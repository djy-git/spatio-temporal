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

Accuracy:  47.1944%

 	 RMSE: 26.986341898329115, MAE: 19.839293571664427

 --- Overall Score --- 
	23.41281773499677

```


# 2. [proposed1.py](proposed1.py)
[baseline1](#1.-baseline1.ipynb)에서 다음을 추가

## Training
- Validation set 추가
- Early stopping 추가

```
 File Name : 
	proposed1.csv

Accuracy:  56.0075%

 	 RMSE: 25.560458459417383, MAE: 19.265550548203443

 --- Overall Score --- 
	22.413004503810413

```


# 3. [proposed2.py](proposed2.py)
[proposed1](#2.-proposed1.ipynb)에서 다음을 추가

## Data
- `StandardScaler` 적용

```
 File Name : 
	proposed2.csv

Accuracy:  48.4391%

 	 RMSE: 27.502376249563028, MAE: 21.10862041048538

 --- Overall Score --- 
	24.305498330024204
```


---


# 4. [proposed3.py](proposed3.py)
[baseline3](#3.-proposed2.ipynb)에서 다음을 추가

## Modeling
각 `TurbID`별로 model을 생성
