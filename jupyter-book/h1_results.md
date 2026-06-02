# H1. CNC 밀링머신 진동데이터셋 — 이상탐지 결과

## 데이터셋

| 항목 | 내용 |
|------|------|
| 데이터셋 | CNC Machining Data (Kaggle) |
| 데이터 | 3축 가속도계 (acc_x, acc_y, acc_z), 2 kHz 샘플링 |
| 정상 (M01) | label_00, label_01, label_02 → 17개 윈도우 |
| 혼합 (M02) | label_11, label_12 → 15개 윈도우 |
| 이상 (M03) | label_20~26 → 100개 윈도우 |
| 윈도우 크기 | 10초 × 2000Hz = 20,000 샘플 |

## 전처리

| 단계 | 내용 |
|------|------|
| 다운샘플링 | 평균 풀링 factor=10 → 20,000 → 2,000 포인트 |
| 특징 추출 | 시간 도메인 27개 + 주파수 도메인 15개 = **총 42개** |
| 정규화 | 특징: MinMaxScaler, Autoencoder: 샘플별 Min-Max |

## 모델 성능 비교

| 모델 | 접근 방식 | 학습 데이터 | F1 Score (weighted) |
|------|-----------|-------------|---------------------|
| Isolation Forest | 특징 기반 (42차원) | 정상 + 이상 | **0.4000** |
| Autoencoder | 재구성 오차 (6000차원) | 정상만 | **0.9440** |

## Autoencoder 상세 결과

| 지표 | Normal | Anomaly |
|------|--------|---------|
| Precision | 1.00 | 0.94 |
| Recall | 0.65 | 1.00 |
| F1-Score | 0.79 | 0.97 |

- 임계값: IQR 방식 (Q3 + 1.5 × IQR)
- 아키텍처: 6000 → 512 → 128 → 64 → 32 (bottleneck) → 64 → 128 → 512 → 6000
- 파라미터 수: 6,302,992
- 학습: 50 epoch, batch_size=32, loss=MAE

## 생성된 시각화

| 파일 | 설명 |
|------|------|
| h1_eda_raw.png | 원시 진동 데이터 파형 비교 (정상 vs 이상) |
| h1_eda_distribution.png | 가속도 축별 KDE 분포 비교 |
| h1_feature_distribution.png | 주요 특징 분포 (acc_z_std, acc_z_rms 등) |
| h1_ae_training.png | Autoencoder 학습 곡선 (Train/Val Loss) |
| h1_threshold.png | 재구성 오차 분포 및 임계값 |
| h1_confusion_matrix.png | 혼동 행렬 |
| h1_reconstruction_error.png | 샘플별 재구성 오차 시각화 |

## 핵심 인사이트

- **Autoencoder가 Isolation Forest를 크게 상회** (F1: 0.94 vs 0.40)
- 정상 데이터로만 학습했음에도 이상 탐지 성능이 우수
- Autoencoder의 Normal Recall이 0.65로 다소 낮은 것은 학습 정상 샘플(17개)이 적기 때문
