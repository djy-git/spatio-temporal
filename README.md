# 2022 인하 인공지능 챌린지 | Spatio-Temporal

> **대회 일정**
> 1. 대회 기간: 22. 6. 20. ~ 22. 7. 22.
> 2. 본 대회: 22. 7. 29. 14:00 ~ 18:00
>    1. Test 데이터셋 공개: 14:00
>    2. 리더보드 운영: 14:00 ~ 18:00
>    3. 소스코드 및 발표자료 제출: 18:00 까지
> 3. 순위발표: 22. 8. 5. 17:00
> 4. 발표 및 시상식: 22. 8. 11. 13:00 ~ 18:00


# 1. [데이터 설명](https://dacon.io/competitions/official/235926/talkboard/406431?page=1&dtype=recent)
```
TurbID  - 발전기 ID
Day     - 날짜
Tmstamp - 시간
Wspd    - 풍속
Wdir    - 터빈이 바라보는 각도와 실제 바람 방향 각도 차이
Etmp    - 외부 온도
Itmp    - 터빈 내부 온도
Ndir    - 터빈이 바라보는 방향 각도
Pab     - 터빈 당 3개의 날이 있으며 각각의 각도가 다름
Prtv    - 무효전력 : 에너지원을 필요로 하지 않는 전력
Patv    - 유효전력 : 실제로 터빈을 돌리는 일을 하는 전력
```

# 2. [평가](https://dacon.io/competitions/official/235926/overview/rules)
## 2.1 평가 산식 및 평가 규제
![](https://dacon.s3.ap-northeast-2.amazonaws.com/competition/235926/editor-image/1656482296560399.jpeg)
```
- 특정 시간에 측정되지 않은 경우, 해당 Patv에 대하여 Error = 0
- 특정 시간에 측정된 Feature의 값이 Patv ≤ 0 and Wspd > 2.5인 경우, Patv 에 대하여 Error = 0
- 특정 시간에 측정된 Feature의 값이 Pab1 > 89° or Pab2 > 89° or Pab3 > 89°인 경우, Patv에 대하여 Error = 0
- 특정 시간에 측정된 Feature의 값이 Ndir > 720° or Ndir < -720°인 경우 Patv에 대하여 Error = 0
- 특정 시간에 측정된 Feature의 값이 Wdir > 180° or Wdir < -180°인 경우, Patv에 대하여 Error = 0

(※ Error = 실제 값 - 예측 값)
```
