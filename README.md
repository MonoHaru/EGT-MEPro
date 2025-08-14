# Marine Engine Fault Diagnostics and Prognostics using Exhaus Gas Temperature data

Official implementation of:

📄 [Marine Engine Fault Diagnostics and Prognostics using Exhaus Gas Temperature data](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11964687)

📰 KIIE (Korean Institute of Industrial Engineers), 2024

[\[PDF\]](src\DBPIA-NURIMEDIA.pdf) [\[POSTER\]](src\poster.pdf)


## 🧑‍🤝‍🧑 Authors

**First Author**
- [Tae-Gyeong Kim](https://github.com/MonoHaru) Department of Artificial Intelligence, Sejong University, Seoul 05006, Republic of Korea, [ktk23114418@sju.ac.kr](mailto:ktk23114418@sju.ac.kr)

**Second Author**
- [Se-Ha Kim](https://github.com/), Department of Artificial Intelligence, Sejong University, Seoul 05006, Republic of Korea, [23114417@sju.ac.kr](mailto:23114417@sju.ac.kr)

**Corresponding Author**
- [Chang-Jae Chun](https://github.com/), Department of Data Science and Artificial Intelligence, Sejong University, Seoul 05006, Republic of Korea, [cchun@sejong.ac.kr](mailto:cchun@sejong.ac.kr)


## 💡 Abstract

##### Early detection and management of faults in a ship’s main engine are essential for ensuring both efficient and safe operation. Such faults can result in substantial economic losses and heightened safety risks. However, conventional fault diagnosis methods are inherently limited, as they can only address issues that have already occurred at the current time point. In this study, we extend fault diagnosis from the present to future time points by utilizing exhaust gas temperature data collected under actual operating conditions. Simulation experiments are conducted to evaluate the feasibility of this approach for main engine fault diagnosis. The research proceeds in three stages. First, various machine learning models are applied to current-time fault diagnosis to determine the most suitable model. Second, a Transformer-based model is developed to predict future time-series data, and its performance is assessed. Finally, fault diagnosis at future time points is performed using the predicted time-series data. Experimental results demonstrate that the proposed approach, leveraging predicted time-series data, achieves a fault diagnosis accuracy of 78.160%.

##### Keywords: Fault diagnosis; Time-series prediction; Marine systems; Exhaust gas temperature; Machine Learning; Transformer


## 📁 Datasets
1. `temp_and_gps_1.csv`
2. `temp_and_gps_2.csv`
3. `temp_and_gps_3.csv`
4. `temp_and_gps_4.csv`
5. `temp_and_gps_5.csv`
6. `temp_and_gps_6.csv`
7. `temp_and_gps_7.csv`
8. `temp_and_gps_8.csv`


## 🚀 Train
| Step        | Description                                      | File                                           |
|-------------|--------------------------------------------------|-----------------------------------------------|
| First Step  | Fault Diagnostics using Machine Learning         | `first_run_fault_diagnosis.ipynb`              |
| Second Step | Time-Series Prediction using Transformer         | `second_run_time_series_prediction.ipynb`      |
| Last Step   | Fault Prognostics (includes Step 1 and Step 2)   | `run.ipynb`                                    |


**Note**

If you wish to perform fault diagnostics only at the current time point, you need to complete only the **First Step**. To predict time-series data for conducting fault diagnosis at a future time point, proceed with the **Second Step**. For fault prognostics at a future time point, carry out the **Last Step**, which includes both the **First Step** and the **Second Step**.


## 📜 License
The code in this repository is released under the MIT License.


## 📖 BibTex
```
@article{김태경2024배기가스,
  title={배기가스 온도 데이터를 활용한 선박 메인엔진 결함 진단 및 예지},
  author={김태경 and 김세하 and 전창재},
  journal={대한산업공학회 추계학술대회 논문집},
  pages={2035--2039},
  year={2024}
}
```