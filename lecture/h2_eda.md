# NASA C-MAPSS Turbofan Engine Degradation — EDA Report

## 1. Dataset Overview

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) 데이터셋은 항공기 터보팬 엔진의 성능 저하를 시뮬레이션한 데이터입니다. 4개의 하위 데이터셋(FD001–FD004)으로 구성되어 있으며, 각각 운영 조건(Operating Conditions) 수와 고장 모드(Fault Mode)가 다릅니다.

| Dataset | Train Samples | Train Engines | Test Engines | Op. Conditions | Fault Mode | Avg Life (cycles) |
|---------|--------------|---------------|--------------|----------------|------------|-------------------|
| FD001   | 4,163        | 100           | 100          | 1              | HPC degradation | 204 |
| FD002   | 10,854       | 260           | 259          | 6              | HPC degradation | 205 |
| FD003   | 4,985        | 100           | 100          | 1              | HPC+Fan degradation | 245 |
| FD004   | 12,346       | 249           | 248          | 6              | HPC+Fan degradation | 244 |

### Features (26 columns)
- **unit**: 엔진 ID
- **cycle**: 운영 사이클 번호 (시간 순서)
- **op_1, op_2, op_3**: 3개의 운영 설정 (Operating Settings)
- **sensor_1 ~ sensor_21**: 21개의 센서 측정값

### Data Structure
- **Train**: 엔진이 완전히 고장 날 때까지의 전체 run-to-failure 데이터
- **Test**: 엔진이 고장 나기 전 어느 시점까지만 제공 (마지막 시점의 RUL을 예측해야 함)
- **RUL**: Test 세트 각 엔진의 실제 Remaining Useful Life (정답)

## 2. Engine Lifetime Distribution

### FD001 (기준 데이터셋)
- **Lifetime range**: 126 ~ 361 cycles
- **Mean**: 204.2 cycles, **Median**: 196.0 cycles
- 엔진 간 수명 편차가 큼 (약 2.9배 차이)

### Test RUL
| Dataset | Min | Max | Mean |
|---------|-----|-----|------|
| FD001   | 7   | 145 | 75.5 |
| FD002   | 6   | 194 | 81.2 |
| FD003   | 6   | 145 | 75.3 |
| FD004   | 6   | 195 | 86.6 |

## 3. Operating Conditions Analysis

### 단일 운영 조건 (FD001, FD003)
- `op_1`, `op_2`: 거의 0에 가까운 고정값 (std ≈ 0)
- `op_3`: 항상 100.0
- 운영 조건이 일정하므로 센서 변화는 엔진 성능 저하에서만 비롯됨

### 다중 운영 조건 (FD002, FD004)
- `op_1`: [0.00, 42.01], 535+ unique values → 연속적 변동
- `op_2`: [0.00, 0.84], 105 unique values
- `op_3`: 2개 값 (60.0, 100.0) → 이산적
- 센서 값이 운영 조건과 성능 저하 모두에 영향을 받음 → 정규화 필수

## 4. Sensor Analysis

### 4.1 Constant Sensors (제외 대상)

센서 값의 표준편차가 0.01 미만인 센서들은 정보량이 없으므로 제외합니다.

| Dataset | Constant Sensors |
|---------|-----------------|
| FD001   | sensor_1, 5, 6, 10, 16, 18, 19 (7개) |
| FD002   | sensor_16 (1개) |
| FD003   | sensor_1, 5, 10, 16, 18, 19 (6개) |
| FD004   | sensor_16 (1개) |

- **sensor_16**은 모든 데이터셋에서 상수 → 항상 제외
- FD001/FD003(단일 운영조건)에서 상수 센서가 많음 → 운영조건 변동 없이는 측정값 변화가 없는 센서들

### 4.2 Low Variance Sensors (std < 1.0)

| Dataset | Low Variance Sensors |
|---------|---------------------|
| FD001   | sensor_2, 7, 8, 11, 12, 13, 15, 20, 21 (9개) |
| FD002   | sensor_10, 15 (2개) |
| FD003   | sensor_2, 6, 8, 11, 13, 15, 20, 21 (8개) |
| FD004   | sensor_10, 15 (2개) |

### 4.3 Top Varying Sensors

| Dataset | Top 6 Sensors by Std |
|---------|---------------------|
| FD001/FD003 | sensor_9, 14, 4, 3, 17, 7 |
| FD002/FD004 | sensor_9, 18, 8, 7, 12, 13 |

**공통 최대 분산 센서**: sensor_9 (Physical Core Speed) — 모든 데이터셋에서 std가 가장 높음

## 5. Sensor-RUL Correlation (FD001)

RUL과의 상관관계가 높은 센서들은 성능 저하를 잘 반영합니다.

### High Positive Correlation (RUL이 클수록 값이 큼 = 정상일 때 높음)
| Sensor | Correlation |
|--------|------------|
| sensor_12 | +0.669 |
| sensor_7  | +0.655 |
| sensor_21 | +0.643 |
| sensor_20 | +0.623 |

### High Negative Correlation (RUL이 작을수록 값이 큼 = 고장 임박시 높음)
| Sensor | Correlation |
|--------|------------|
| sensor_11 | -0.702 |
| sensor_4  | -0.677 |
| sensor_15  | -0.641 |
| sensor_17 | -0.613 |
| sensor_2  | -0.594 |
| sensor_3  | -0.564 |
| sensor_8  | -0.564 |
| sensor_13 | -0.561 |

### 해석
- sensor_11이 RUL과 가장 강한 음의 상관관계 (-0.70) → 고장 임박 시 값이 증가
- 양의 상관 센서들은 정상 상태에서 높은 값을 가지다가 고장에 가까워지면 감소하는 패턴

## 6. Degradation Pattern (FD001)

RUL 구간별 주요 센서 평균값:

| RUL Range | sensor_9 | sensor_14 | sensor_4 | sensor_11 | sensor_12 |
|-----------|----------|-----------|----------|-----------|-----------|
| 0-50      | 9078.83  | 8153.20   | 1418.21  | (↑ high)  | 520.69    |
| 51-100    | 9063.51  | 8142.23   | 1408.32  | (↑)       | 521.46    |
| 101-150   | 9058.51  | 8138.98   | 1404.82  | (→)       | 521.76    |
| 151-200   | 9056.77  | 8138.00   | 1403.24  | (→)       | 521.89    |
| 201-250   | 9057.14  | 8138.92   | 1402.25  | (→)       | 521.98    |
| 251-300   | 9058.18  | 8139.71   | 1400.54  | (→)       | 522.05    |
| 300+      | 9055.51  | 8136.95   | 1402.76  | (→)       | 522.02    |

### 관찰
1. **sensor_9, 14, 4**: RUL이 감소할수록 (고장에 가까워질수록) 값이 증가 → 명확한 노화 패턴
2. **sensor_12**: RUL이 감소할수록 값이 감소 → 반대 방향 노화 패턴
3. **RUL > 200**: 센서 값이 거의 일정 → 엔진이 아직 노화되지 않은 구간
4. **RUL < 100**: 센서 변화가 가속화됨 → 급격한 성능 저하 구간

## 7. RUL Clipping

RUL 상한선을 125로 자르는(clipping) 것은 NASA C-MAPSS 연구의 표준 전처리입니다.

| Metric | Unclipped | Clipped (125) |
|--------|-----------|---------------|
| Mean   | 106.7     | 86.0          |
| Std    | 69.5      | 42.4          |
| Samples > 125 | 1,563 (37.5%) | 0 |

### 클리핑 이유
1. **초기 구간 일정화**: RUL이 125 이상인 구간은 엔진이 아직 건강하여 센서에 노화 신호가 거의 없음
2. **학습 안정화**: 타겟 분산을 줄여 모델 학습이 안정적이 됨
3. **현실적 의미**: 실제 현장에서는 RUL=200과 RUL=300의 차이보다, RUL=10과 RUL=50의 차이가 훨씬 중요함

## 8. Preprocessing Recommendations

### 기본 전처리 파이프라인
1. **Constant sensor 제거**: FD001의 경우 sensor_1, 5, 6, 10, 16, 18, 19 제거 → 14개 센서 사용
2. **RUL 라벨 생성**: `max_cycle - current_cycle`
3. **RUL 클리핑**: `rul = min(rul, 125)`
4. **정규화**: Min-Max 또는 Standard Scaler (센서별)
5. **Sliding window**: LSTM 입력을 위해 시퀀스 길이 30~50 사용

### 센서 선택 전략
- **FD001/FD003**: 상수 센서 6-7개 제거 → 14-15개 센서 사용
- **FD002/FD004**: sensor_16만 제거 + 운영조건 정규화 → 20개 센서 + 3개 op 사용
- **선택적**: RUL 상관관계 |r| > 0.5인 센서만 사용 → 14개 (FD001)

### Autoencoder용 vs LSTM용
- **Autoencoder**: 정상 구간(RUL > 125) 데이터로만 학습 → 재구성 오차로 이상 탐지
- **LSTM RUL**: 전체 데이터 + RUL 타겟으로 회귀 학습 → 직접 RUL 예측

## 9. Key Takeaways

1. **FD001이 교육용 기준**: 단일 운영조건, 가장 단순, 가장 많이 연구됨
2. **14개 유효 센서**: FD001에서 21개 중 7개 상수 → 14개만 사용
3. **명확한 노화 패턴**: RUL이 감소할수록 센서 값이 체계적으로 변화
4. **RUL 클리핑(125)**: 학습 안정성과 예측 정확도 향상을 위한 핵심 전처리
5. **운영조건 영향**: FD002/FD004는 운영조건 변동이 크므로 정규화가 필수
6. **센서-RUL 상관관계**: |r| > 0.6인 센서가 8개 → 충분한 예측력 보유
