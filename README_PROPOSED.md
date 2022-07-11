# 1. [baseline1.ipynb](proposed/baseline1.ipynb)
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

Accuracy:  46.7085%

 	 RMSE: 26.80773266298162, MAE: 19.66573268982056

 --- Overall Score --- 
	23.23673267640109
```


# 2. [proposed1.ipynb](proposed/proposed1.ipynb)
**baseline1** 에서 다음을 추가

## Data
- `StandardScaler` 적용

```
 File Name : 
	proposed1.csv

Accuracy:  47.1609%

 	 RMSE: 50.91567351662873, MAE: 46.670559504018996

 --- Overall Score --- 
	48.79311651032386
```

---

# 3. [proposed2.ipynb](proposed/proposed2.ipynb)
**baseline1** 에서 다음을 추가

## Training
- Validation set 추가
- Early stopping 추가

