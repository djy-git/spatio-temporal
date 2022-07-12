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
- Early stopping(patience=10) 추가
    - Overfitting이 심해 많이 학습하지 못한다 → Regularization 필요

```
 File Name : 
	proposed2.csv

Accuracy:  54.0560%

 	 RMSE: 24.517586741073245, MAE: 17.771835530432146

 --- Overall Score --- 
	21.144711135752694
```


# 4. [proposed3.ipynb](proposed/proposed3.ipynb)
**proposed2** 에서 다음을 추가

## Model
- `BatchNormalization`, `Dropout` 추가
  ```
  model = Sequential([
      GRU(256, input_shape=train_x[0].shape),
      BatchNormalization(),
      Dropout(0.4),
      
      Dense(516, activation='relu'),
      BatchNormalization(),
      Dropout(0.4),
  
      Dense(288, activation='relu')
  ])
  ```

```
 File Name : 
	proposed3.csv

Accuracy:  40.8675%

 	 RMSE: 26.95957636335696, MAE: 19.47166986231859

 --- Overall Score --- 
	23.215623112837775
```


# 5. [proposed4.ipynb](proposed/proposed4.ipynb)
**proposed2** 에서 다음을 추가

## Data
- `TurbID` 별로 model을 생성

## Training
- `batch_size=32`

```
 File Name : 
	proposed4.csv

Accuracy:  53.8061%

 	 RMSE: 29.735404515681957, MAE: 24.527271051411567

 --- Overall Score --- 
	27.131337783546762
```


# 5. [proposed5.ipynb](proposed/proposed5.ipynb)
**proposed2** 에서 다음을 추가

## Data
- Missing value 처리
  - `Day=65, 66, 67` 제거
  - `Day=1, Tmstamp=00:00` 
  - 나머지는 interpolation 사용하여 채우기
  - e.g. `... → [(58, 59, 60, 61, 62), (63, 64)] → [(68, 69, 70, 71, 72), (73, 74)] → ...`


```
 File Name : 
	proposed5.csv

Accuracy:  50.4010%

 	 RMSE: 25.910936738551044, MAE: 19.427390262693685

 --- Overall Score --- 
	22.669163500622364
```

---

# 6. [proposed6.ipynb](proposed/proposed6.ipynb)
**proposed5** 에서 다음을 추가

## Data
- `Day` 대신 `Tmstamp` 단위로 데이터 분할


---

1. 기존의 잘 나왔던 연구들이나 코드들을 보고 따라하기
   - Data
   - Model
3. Normalization features
4. 학습이 잘 안 된다.
   - 
