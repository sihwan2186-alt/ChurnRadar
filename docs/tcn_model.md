# TCN 모델 추가

## Temporal Convolutional Network 기반 경량 시계열 이탈 탐지 모델

ChurnRadar는 고객 이탈 징후를 탐지하기 위해 정적 머신러닝 모델과 시계열 딥러닝 모델을 함께 비교한다.

기존 비교 흐름은 다음과 같았다.

```text
XGBoost -> LSTM -> TS-Transformer
```

여기에 TCN(Temporal Convolutional Network)을 추가하여 비교 구조를 다음처럼 확장한다.

```text
XGBoost -> LSTM -> TCN -> TS-Transformer
```

## TCN을 추가하는 이유

TCN을 추가하는 이유는 단순히 모델 개수를 늘리기 위해서가 아니다. 본 프로젝트는 고객의 30일 행동 데이터를 기반으로 이탈 징후를 탐지하며, 하루 단위의 사용량보다 여러 날에 걸쳐 나타나는 사용량 감소, 모멘텀 하락, 가속도 변화가 중요하다.

TCN은 1D Convolution을 사용해 일정 구간의 변화 패턴을 병렬적으로 학습한다. 따라서 고객 행동 데이터를 하루씩 순서대로 읽기보다, 여러 날짜에 걸쳐 나타나는 패턴을 묶어서 탐지할 수 있다.

## 입력 데이터 구조

TCN은 기존 시계열 파이프라인에서 생성한 3D Tensor 데이터를 그대로 사용한다.

```text
Input Shape = Customer x 30 Days x 3 Features
```

| Feature | 의미 |
| --- | --- |
| Energy | 현재 고객의 절대적인 사용량 |
| Momentum | 전날 대비 사용량 변화 속도 |
| Acceleration | 사용량 감소 속도의 변화량 |

기존 `ChurnTimeSeriesDataset`과 TS-SMOTE 흐름을 재사용하므로 별도 Tensorization 파이프라인을 새로 만들 필요가 없다.

## LSTM과의 차이

LSTM은 30일 데이터를 시간 순서대로 하나씩 읽는다.

```text
1일차 -> 2일차 -> 3일차 -> ... -> 30일차
```

이 방식은 시간 흐름을 반영할 수 있지만, 초반에 나타난 미세한 이탈 징후가 후반부로 갈수록 약해질 수 있다.

TCN은 1D Convolution과 dilation을 사용하여 여러 시점의 구간 패턴을 함께 본다.

```text
1~3일 패턴
4~6일 패턴
10~15일 패턴
20~30일 패턴
```

따라서 사용량이 서서히 줄어드는 패턴이나 특정 시점 이후 급격히 꺾이는 패턴을 안정적으로 포착할 수 있다.

## Transformer와의 차이

TS-Transformer는 Self-Attention으로 30일 전체 데이터에서 중요한 날짜나 변화 구간에 집중한다. 강력하지만 상대적으로 연산량이 크다.

TCN은 모든 시점 간 관계를 정교하게 계산하지는 않지만, Convolution 기반 구조라 더 가볍고 빠르게 동작할 수 있다. 그래서 빠른 추론, 대량 고객 배치 스코어링, 서버 비용 절감이 필요한 운영 환경에서 비교 가치가 있다.

| 모델 | 역할 | 특징 |
| --- | --- | --- |
| XGBoost | 정적 기준 모델 | 현재 상태만 보고 이탈 판단 |
| LSTM | 순차 시계열 모델 | 30일 데이터를 순서대로 학습 |
| TCN | 경량 시계열 모델 | 1D Convolution으로 변화 구간 탐지 |
| TS-Transformer | Attention 기반 모델 | 중요한 날짜와 징후에 집중 |

## 프로젝트 내 역할

TCN은 최종 모델을 무조건 대체하기 위한 모델이 아니라, LSTM과 TS-Transformer 사이의 중간 비교 모델이다.

이를 통해 다음 비교가 가능하다.

- 정적 머신러닝 모델과 시계열 모델의 차이
- LSTM과 TCN의 시계열 패턴 탐지 방식 차이
- TCN과 TS-Transformer의 성능 및 연산 효율 차이
- 최종 모델 선택 근거 강화

## 구현 위치

- 모델 클래스: `src/models/tcn_model.py`
- 학습 스크립트: `scripts/train_tcn.py`
- 기본 저장 경로: `models/churn_tcn.pth`
- 학습 요약 저장 경로: `results/tcn_training_summary.json`
- API 단독 예측 엔드포인트: `POST /predict/tcn`

학습 실행 예시는 다음과 같다.

```bash
python scripts/train_tcn.py --epochs 50 --batch_size 512
```

## 한 줄 요약

TCN은 LSTM의 시계열 기억 한계를 보완하고, TS-Transformer보다 가벼운 방식으로 30일 행동 변화 패턴을 안정적으로 탐지하기 위한 경량 시계열 비교 모델이다.
