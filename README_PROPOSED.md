# 1. [baseline1.ipynb](baseline1.ipynb): 22
## Data
- Train data: 5일
- Test data: 2일
- Sliding window로 생성

## Training
- NN: GRU(256) - Dense(516) - Dense(288)
- Optimizer: RMSProp(lr=1e-3)
- Loss: MSE
- Epochs: 10
- Batch size: 128

```
 File Name : 
	baseline1.csv

Accuracy:  68.5368%

 	 RMSE: 57.690916663169546, MAE: 47.12849869928885

 --- Overall Score --- 
	52.40970768122919
```


# 2. [proposed1.py](proposed1.py)
[baseline1](#1.-baseline1.ipynb)에서 다음을 추가

## Training
- Validation set 추가
- Early stopping 추가

```
 File Name : 
	proposed1.csv

Accuracy:  71.5641%

 	 RMSE: 50.30975422526733, MAE: 40.59707643071805

 --- Overall Score --- 
	45.45341532799269
```


# 3. [proposed2.py](proposed2.py)
[proposed1](#2.-proposed1.ipynb)에서 다음을 추가

## Data
- `StandardScaler` 적용

```
 File Name : 
	proposed2.csv

Accuracy:  67.7897%

 	 RMSE: 47.04977466266756, MAE: 35.562956344972264

 --- Overall Score --- 
	41.30636550381991
```


---


# 4. [proposed3.py](proposed3.py)
[baseline3](#3.-proposed2.ipynb)에서 다음을 추가

## Modeling
각 `TurbID`별로 model을 생성
