# H2. NASA Turbofan Engine — RUL(잔여수명) 예측 결과

## 데이터셋

| 항목 | 내용 |
|------|------|
| 데이터셋 | NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) |
| 하위셋 | FD001 (단일 운영조건 + 단일 고장모드) |
| 학습 데이터 | 100대 엔진, 4,163행, Run-to-Failure |
| 테스트 데이터 | 100대 엔진, 2,663행 |
| RUL 정답 범위 | 7 ~ 145 사이클 |
| 컬럼 | unit, cycle, 3개 op_setting, 21개 센서 |

## 전처리

| 단계 | 내용 |
|------|------|
| 상수 센서 제거 | 표준편차 < 0.5 → **14개 제거**, 7개 유효 센서 사용 |
| RUL 라벨 | Linear RUL + Clipped RUL (cap=125) |
| 정규화 | MinMaxScaler (fit: train, transform: test) |
| Sliding Window | SEQUENCE_LENGTH=30, stride=1 → 1,270개 학습 시퀀스 |
| 시퀀스 형태 | (samples, 30, 7) |

## 제거된 상수 센서 (14개)

sensor_1, sensor_2, sensor_5, sensor_6, sensor_8, sensor_10, sensor_11, sensor_13, sensor_15, sensor_16, sensor_18, sensor_19, sensor_20, sensor_21

## 사용된 유효 센서 (7개)

sensor_3, sensor_4, sensor_7, sensor_9, sensor_12, sensor_14, sensor_17

## 모델 아키텍처

| 레이어 | 설정 | 파라미터 |
|--------|------|----------|
| LSTM(64) | return_sequences=True | 18,432 |
| Dropout(0.2) + BatchNorm | — | 256 |
| LSTM(32) | return_sequences=False | 12,416 |
| Dropout(0.2) + BatchNorm | — | 128 |
| Dense(64, ReLU) | — | 2,112 |
| Dropout(0.2) | — | — |
| Dense(32, ReLU) | — | 2,080 |
| Dense(1, linear) | RUL 회귀 출력 | 33 |
| **총계** | | **35,457** |

## 클리핑 효과 비교

| 타겟 | RMSE | NASA S-Score |
|------|------|-------------|
| Linear RUL | **45.75** | **975.32** |
| Clipped RUL (cap=125) | 57.97 | 31,829.86 |

> 이번 실행에서는 Linear RUL이 더 낮은 RMSE를 기록했으나, 일반적으로 캐글 메달 노트북에서는 Clipped RUL이 더 안정적인 성능을 보임. 결과는 데이터셋 크기와 하이퍼파라미터에 따라 달라질 수 있음.

## 학습 과정

| Epoch | Train Loss (MSE) | Train MAE | Val Loss | Val MAE |
|-------|-------------------|-----------|----------|---------|
| 1 | 2,794.74 | 40.57 | 4,771.34 | 55.02 |
| 10 | 1,473.06 | 29.73 | 3,331.01 | 43.68 |
| 25 | 234.55 | 11.69 | 1,624.07 | 32.55 |
| 50 | 132.56 | 8.63 | 1,492.54 | 30.32 |

## 생성된 시각화

| 파일 | 설명 |
|------|------|
| h2_sensor_trends.png | 21개 센서 트렌드 (빨간 배경=상수 센서) |
| h2_rul_distribution.png | Linear vs Clipped RUL 분포 |
| h2_rul_comparison.png | 샘플 엔진 RUL 곡선 비교 |
| h2_training_curves.png | Loss, MAE, S-Score 학습 곡선 |
| h2_rul_prediction.png | 예측 vs 실제 산점도 + 엔진별 바차트 |
| h2_error_analysis.png | 오차 분포 + 오차 크기 vs True RUL |
| h2_clipping_comparison.png | Linear vs Clipped RUL 예측 비교 |

## 핵심 인사이트

- 21개 센서 중 14개가 상수/준상수로 제거 → **7개 유효 센서만으로 예측 가능**
- NASA S-Score: 비대칭 페널티로 **늦은 예측에 더 큰 벌점** 부여
- Sliding Window (stride=1)로 데이터 증강 효과: 100대 엔진 → 1,270개 학습 샘플
- 짧은 수명 엔진 4대(26~29사이클)는 시퀀스 길이 미달로 자동 제외
