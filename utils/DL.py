import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from tqdm.notebook import tqdm
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings('ignore')

# Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout = 0.5):
        super(PositionalEncoding, self).__init__()   
        self.dropout = nn.Dropout(p = dropout)    
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.5):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout = dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    

# Dataset
class TimeSeiresDataset(Dataset):
    def __init__(self, X,y, input_window):
        self.input_window = input_window
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx].type(torch.float32), self.y[idx].type(torch.float32)
    
    def __len__(self):
        return len(self.X)
    

# Data Preprocessing
def multistep_time_series(temp_data, label_data, input_window, output_window):
    inout_seq = []
    label = []
    batch_len= input_window + output_window
    L = len(temp_data)
    for i in range(L-batch_len):
        train_seq = temp_data[i:i+input_window]
        train_label = temp_data[i+output_window:i+input_window+output_window]
        temp_label = max(label_data[i+output_window:i+input_window+output_window])
        
        inout_seq.append((train_seq ,train_label))
        label.append(temp_label)
    return torch.FloatTensor(inout_seq), label

def get_batch(source, i,batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target

calculate_loss_over_all_values =  False

def train_tmp(model, train_data,batch_size, optimizer, criterion, input_window, output_window, epoch, scheduler):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size, input_window)
        optimizer.zero_grad()
        output = model(data)        

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f} |'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Train
def train(model, train_dataloader, device, optimizer, criterion, epoch, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.0
    
    for idx, batch in enumerate(train_dataloader):
        input, label = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_dataloader)  / 5)
        
        if idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('|epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | loss {:5.5f}'.format(epoch, idx, len(train_dataloader),
                                                                                           scheduler.get_lr()[0], cur_loss ))
            total_loss = 0
            start_time = time.time()


# Validation
def calculate_loss_and_plot(model, test_dataloader, device, criterion, output_window, scaler_test,batch_size):
    model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    result_to_ML = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):
            input_batch, label_batch = batch[0].to(device), batch[1].to(device)
            
            for jdx in range(input_batch.shape[0]):
                input = input_batch[jdx].unsqueeze(0)
                label = label_batch[jdx].unsqueeze(0)
                output = model(input)
                loss = criterion(output, label)
                total_loss += loss.item()
                test_result = torch.cat((test_result, output[:, -output_window:].view(-1).cpu()), 0) #todo: check this. -> looks good to me
                truth = torch.cat((truth, label[:, -output_window:].view(-1).cpu()), 0)
                result_to_ML.append(output[:, -output_window:].view(-1).cpu().detach().numpy())
    
    test_result = scaler_test.inverse_transform(test_result.reshape(-1,1)).reshape(-1)
    truth = scaler_test.inverse_transform(truth.reshape(-1,1)).reshape(-1)
    
    plt.plot(test_result, label = 'prediction')
    plt.plot(truth, label = 'truth')
    plt.grid(True, which = 'both')
    plt.ylim([500,1000])
    plt.axhline(y=0, color='k')
    plt.show()
    plt.close()
    

    return truth, test_result, result_to_ML, total_loss / idx

def new_multistep_time_series(temp_data, label_data, input_window, output_window):
    inout_seq = []
    label = []
    batch_len = input_window + output_window
    L = len(temp_data)
    for i in range(L-batch_len):
        train_seq = temp_data[i : i + input_window]
        train_label = temp_data[i + output_window : i + output_window + input_window] #[40 : ]
        min_temp_label = min(label_data[i + output_window : i+output_window+input_window])
        max_temp_label = max(label_data[i + output_window : i+output_window+input_window])
        
        if min_temp_label == max_temp_label:
            inout_seq.append((train_seq, train_label))
            label.append(max_temp_label)
        else:
            continue
            
    return torch.FloatTensor(inout_seq), label

def time_series_dataframe_ML():
    path_temp_gps = './temp_add_gps/'
    list_temp_gps = os.listdir(path_temp_gps)

    m1 = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[0]))
    for i in range(1,8):
        tmp = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[i]))
        m1 = pd.concat([m1, tmp], axis = 0)

    m1 = m1.reset_index(drop = True)
    m1 = m1[m1['TEMP']>=243.07].reset_index(drop = True)
    time_df = m1
    time_df = time_df.loc[:, ['TEMP']]

    for i in range(1,8):
        globals()['df_'+str(i)+'_temp'] = time_df[60436*(i-1):60436*i].reset_index(drop = True)

    N = 6
    dx = (600 - df_1_temp['TEMP'].mean()) / N # ??? ??????? ???? ?????? : 56.3785
    dx_minute = dx / (len(df_1_temp)-1) # ???? ??????

    time = np.arange(len(df_1_temp))
    slope = dx_minute * 2

def plot_and_loss(model, data_source, criterion,input_window, output_window, scaler_DL):
    model.eval() 
    print("Evaluation...")
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    result_to_ML = []
    with torch.no_grad():
        for i in tqdm(range(len(data_source)-1)):
            data, target = get_batch(data_source, i,1, input_window)
            # look like the model returns static values for the output window
            output = model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            result_to_ML.append(output[-output_window:].view(-1).cpu().detach().numpy())
            
    test_result = scaler_DL.inverse_transform(test_result.reshape(-1,1)).reshape(-1)
    truth = scaler_DL.inverse_transform(truth.reshape(-1,1)).reshape(-1)
    
    plt.plot(test_result,label = 'Prediction')
    plt.plot(truth,label = 'Truth')
    #pyplot.plot(test_result-truth,color="green")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    plt.close()
    
    return truth, test_result, result_to_ML, total_loss / i

def plot_and_loss2(model, data_source, criterion,input_window, output_window, scaler_DL):
    model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    result_to_ML = []
    with torch.no_grad():
        for i in tqdm(range(len(data_source)-1)):
            data, target = get_batch(data_source, i,1, input_window)
            # look like the model returns static values for the output window
            output = model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            result_to_ML.append(output[-output_window:].view(-1).cpu().detach().numpy())
            
    test_result = scaler_DL.inverse_transform(test_result.reshape(-1,1)).reshape(-1)
    truth = scaler_DL.inverse_transform(truth.reshape(-1,1)).reshape(-1)
    
    plt.plot(test_result,label = 'Prediction')
    plt.plot(truth,label = 'Truth')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    plt.close()
    
    return truth, test_result, result_to_ML, total_loss / i

def evaluate(model, test_dataloader, device, criterion, output_window):
    model.eval()
    total_loss = 0.0
    batch_size = 512
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input, label = batch[0].to(device), batch[1].to(device)
            output = model(input)
            total_loss += criterion(output[:, -output_window:], label[:, -output_window:]).item()

    return total_loss / (len(test_dataloader) * batch_size)

def evaluate2(model, data_source, criterion, output_window, input_window):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size, input_window)
            output = model(data)            
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)