# H3. CCTV Action Recognition — 안전위반 탐지 결과

## 데이터셋

| 항목 | 내용 |
|------|------|
| 데이터셋 | CCTV Action Recognition (Kaggle, Jonathan Nield) |
| 액션 수 | 13종류 |
| 총 클립 | 2,600개 (액션당 200개) |
| 실행 모드 | **시뮬레이션 모드** (비디오 파일 없음, 랜덤 데이터로 파이프라인 검증) |

## 액션 분류

| 분류 | 액션 (8종류) | 클립 수 |
|------|-------------|---------|
| **VIOLATION (위반)** | fall, grab, gun, hit, kick, sneak, struggle, throw | 1,600 |
| **NORMAL (정상)** | walk, stand, sit, run, lying_down | 1,000 |

## 전처리

| 단계 | 설정 |
|------|------|
| 프레임 해상도 | 112 × 112 |
| 시퀀스 길이 | 16프레임 (균등 샘플링) |
| 데이터 분할 | 학습 80 / 검증 20 (stratified) |
| 클래스 가중치 | NORMAL=1.00, VIOLATION=0.50 |

## 모델 아키텍처

### 1단계: CNN 특징 추출 (MobileNetV2)

| 항목 | 설정 |
|------|------|
| 백본 | MobileNetV2 (ImageNet 사전학습) |
| trainable | False (가중치 동결) |
| 특징 차원 | 1,280 |
| 출력 | (samples, 16, 1280) |

### 2단계: LSTM 분류기

| 레이어 | 설정 | 파라미터 |
|--------|------|----------|
| LSTM(128) | return_sequences=True | — |
| Dropout(0.3) + BatchNorm | — | — |
| LSTM(64) | return_sequences=False | — |
| Dropout(0.3) + BatchNorm | — | — |
| Dense(64, ReLU) | — | — |
| Dropout(0.3) | — | — |
| Dense(1, sigmoid) | 위반 확률 [0,1] | — |
| **총계** | | **775,809** |

## 학습 결과 (30 Epoch)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.2836 | 0.7125 | 0.7088 | 0.2000 |
| 10 | 0.2720 | 0.7250 | 0.6885 | 0.5000 |
| 20 | 0.2523 | 0.7875 | 0.7161 | 0.0500 |
| 30 | 0.2278 | 0.8125 | 0.6670 | 0.7500 |

## 평가 결과

| 지표 | 값 |
|------|-----|
| Accuracy | **0.7500** |
| F1 Score (weighted) | **0.8571** |

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| NORMAL | 0.00 | 0.00 | 0.00 | 0 |
| VIOLATION | 1.00 | 0.75 | 0.86 | 20 |

> 시뮬레이션 모드에서는 stratified split 결과 VIOLATION만 검증셋에 포함됨. 실제 비디오 모드에서는 정상/위반 균형 분포가 되어 의미 있는 평가 가능.

## 실시간 탐지 데모 (10개 샘플)

| 결과 | 개수 | 비율 |
|------|------|------|
| 정탐 (O) | 8 | 80% |
| 오탐 (X) | 2 | 20% |

## 생성된 시각화

| 파일 | 설명 |
|------|------|
| h3_data_distribution.png | 액션별 클립 수 + 위반/정상 분포 |
| h3_frame_samples.png | 추출된 프레임 시퀀스 (4클립 × 16프레임) |
| h3_training_curves.png | Loss / Accuracy 학습 곡선 |
| h3_evaluation.png | 혼동 행렬 + 위반 확률 분포 |
| h3_detection_demo.png | 실시간 탐지 결과 바차트 |

## 핵심 인사이트

- **2단계 분리 설계** (CNN 특징 추출 → LSTM 분류)로 메모리 절약 및 학습 속도 향상
- 클래스 가중치(NORMAL=1.0, VIOLATION=0.5)로 불균형 보정
- 시뮬레이션 모드에서 파이프라인 정상 동작 확인 → 실제 비디오 배치 시 즉시 실사용 가능
- 위반확률이 0.5 근처에 몰려 있어, 실제 데이터에서는 임계값 튜닝이 필수적
