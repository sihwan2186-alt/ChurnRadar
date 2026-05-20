# TCN 모델 추가

## Temporal Convolutional Network 기반 경량 시계열 이탈 탐지 모델

## 1. 도입 상태 확인

TCN은 현재 프로젝트에 이미 도입되어 있다.

구현 위치:

- 모델 클래스: `src/models/tcn_model.py`
- 학습 스크립트: `scripts/train_tcn.py`
- 평가 연동: `scripts/evaluate_pro.py`
- API 예측 연동: `api/model_handler.py`, `api/main.py`
- 테스트: `tests/test_tcn_model.py`
- 기본 모델 저장 경로: `models/churn_tcn.pth`
- 학습 결과 요약: `results/tcn_training_summary.json`
- 비교 결과 요약: `results/tcn_comparison_summary.json`
- API 단독 예측 엔드포인트: `POST /predict/tcn`

즉, 새로 도입해야 하는 상태는 아니며, 본 문서는 TCN을 프로젝트 보고서와 발표에 설명할 수 있도록 정리한 것이다.

## 2. 모델 추가 배경

기존 프로젝트에서는 고객 이탈 징후를 탐지하기 위해 정적 머신러닝 모델과 시계열 딥러닝 모델을 함께 사용한다.

기존 모델 흐름은 다음과 같다.

```text
XGBoost -> LSTM -> TS-Transformer
```

하지만 LSTM과 TS-Transformer 사이에는 성능과 연산 효율 측면에서 비교할 수 있는 중간 모델이 필요하다.

- LSTM은 시간 순서대로 데이터를 읽기 때문에 초기 이탈 징후를 후반부에서 약하게 반영할 수 있다.
- TS-Transformer는 Self-Attention을 통해 중요한 시점을 잘 포착하지만 상대적으로 연산량이 크다.

이를 보완하기 위해 TCN(Temporal Convolutional Network)을 비교 모델로 추가한다.

TCN을 포함한 비교 구조는 다음과 같다.

```text
XGBoost -> LSTM -> TCN -> TS-Transformer
```

## 3. TCN이란?

TCN은 시계열 데이터를 처리하기 위한 딥러닝 모델이다.

LSTM처럼 데이터를 순서대로 하나씩 읽는 방식이 아니라, 1D Convolution을 사용해 여러 시점의 변화 패턴을 병렬적으로 학습한다.

즉, 고객의 30일 행동 데이터를 한 줄씩 순차적으로 읽는 것이 아니라 일정 구간의 사용량 변화 패턴을 빠르게 훑으면서 이탈 징후를 탐지한다.

## 4. 입력 데이터 구조

TCN은 기존 프로젝트에서 생성한 3D Tensor 데이터를 그대로 사용할 수 있다.

```text
Input Shape = Customer x 30 Days x 3 Features
```

3개 feature는 다음과 같다.

| Feature | 의미 |
| --- | --- |
| Energy | 현재 고객의 절대적인 사용량 |
| Momentum | 전날 대비 사용량 변화 속도 |
| Acceleration | 사용량 감소 속도의 변화량 |

기존 3D Tensorization 파이프라인을 수정하지 않고도 TCN 모델을 추가할 수 있다. 현재 `scripts/train_tcn.py`는 parquet 시계열 입력을 우선 사용하고, 없으면 Baza CSV에서 30일 synthetic sequence를 생성해 학습할 수 있다.

## 5. LSTM과의 차이

LSTM은 30일 데이터를 시간 순서대로 하나씩 읽는다.

```text
1일차 -> 2일차 -> 3일차 -> ... -> 30일차
```

이 방식은 시간 흐름을 반영할 수 있다는 장점이 있지만, 초반에 나타난 미세한 이탈 징후가 후반부로 갈수록 약해지거나 잊힐 수 있다.

반면 TCN은 1D Convolution과 dilation을 사용하여 여러 시점의 구간 패턴을 함께 본다.

```text
1~3일 패턴
4~6일 패턴
10~15일 패턴
20~30일 패턴
```

즉, 고객 행동 변화를 하루씩 순서대로 읽는 것이 아니라 일정 기간 동안 나타나는 패턴을 묶어서 탐지한다. 이를 통해 사용량이 서서히 줄어드는 패턴이나 특정 시점 이후 급격히 꺾이는 패턴을 LSTM보다 안정적으로 포착할 수 있다.

## 6. Transformer와의 차이

TS-Transformer는 30일 전체 데이터를 한 번에 보고, Self-Attention을 통해 중요한 날짜나 변화 구간에 집중한다. 이 방식은 강력하지만 상대적으로 연산량이 크다.

반면 TCN은 Transformer만큼 모든 시점 간 관계를 정교하게 계산하지는 않지만, 1D Convolution 기반의 단순한 구조를 사용하기 때문에 더 가볍고 빠르게 동작할 수 있다.

| 모델 | 특징 |
| --- | --- |
| LSTM | 순서대로 읽지만 초기 징후를 약하게 반영할 수 있음 |
| TCN | 구간 패턴을 안정적으로 탐지함 |
| TS-Transformer | 전체 날짜 관계를 보고 중요한 시점에 집중함 |

## 7. 프로젝트 내 역할

TCN은 최종 모델을 무조건 대체하기 위한 모델이 아니다. 오히려 LSTM과 TS-Transformer 사이의 중간 비교 모델 역할을 한다.

```text
XGBoost -> LSTM -> TCN -> TS-Transformer
```

이 구조를 통해 다음과 같은 비교가 가능해진다.

- 정적 머신러닝 모델과 시계열 모델의 차이
- LSTM과 TCN의 시계열 패턴 탐지 방식 차이
- TCN과 TS-Transformer의 성능 및 연산 효율 차이
- 최종적으로 TS-Transformer 또는 ensemble을 선택한 근거 강화

## 8. 모델 비교 구조

| 모델 | 역할 | 특징 |
| --- | --- | --- |
| XGBoost | 정적 기준 모델 | 현재 상태만 보고 이탈 판단 |
| LSTM | 순차 시계열 모델 | 30일 데이터를 순서대로 학습 |
| TCN | 경량 시계열 모델 | 1D Convolution으로 변화 패턴 탐지 |
| TS-Transformer | Attention 기반 모델 | 중요한 날짜와 징후에 집중 |

이를 통해 단순히 하나의 모델만 사용한 것이 아니라, 정적 모델부터 시계열 딥러닝 모델까지 단계적으로 비교했다는 근거를 만들 수 있다.

## 9. TCN 추가 이점

### 9.1 LSTM의 장기 의존성 문제 보완

LSTM은 데이터를 순서대로 읽기 때문에 30일 초반에 나타난 미세한 이탈 징후를 후반부에서 약하게 반영할 수 있다.

반면 TCN은 일정 기간의 패턴을 합성곱으로 한 번에 탐지하므로, 초기 사용량 감소나 모멘텀 하락 패턴을 더 안정적으로 포착할 수 있다.

### 9.2 Transformer보다 가벼운 구조

TS-Transformer는 Self-Attention 기반으로 강력하지만 연산량이 크다.

TCN은 Convolution 기반 모델이므로 상대적으로 가볍고 학습 속도가 빠르다. 따라서 다음 운영 환경에서 장점이 있다.

- 빠른 추론이 필요한 경우
- 매일 대량 고객을 배치 스코어링해야 하는 경우
- 서버 비용을 줄여야 하는 경우
- TS-Transformer의 경량 대체 모델이 필요한 경우

### 9.3 모델 검증 깊이 강화

TCN을 추가하면 발표에서 다음 메시지를 전달할 수 있다.

> 단순히 TS-Transformer 하나만 사용한 것이 아니라, XGBoost, LSTM, TCN, TS-Transformer를 비교하여 어떤 모델 구조가 고객 이탈 징후 탐지에 가장 적합한지 검증했다.

이는 프로젝트의 실험 설계와 모델 검증 완성도를 높여준다.

## 10. 현재 실험 결과

현재 저장된 TCN 학습 요약 기준:

| 항목 | 값 |
| --- | ---: |
| 입력 형식 | csv_synthetic_30day |
| 입력 shape | 8,453 x 30 x 3 |
| 채널 구조 | 32, 64 |
| kernel size | 3 |
| parameter 수 | 26,337 |
| epoch | 20 |
| best validation F1 | 0.1247 |
| best validation recall | 0.9273 |
| best validation precision | 0.0668 |
| best threshold | 0.45 |

`results/tcn_comparison_summary.json` 기준으로는 standalone TCN이 XGBoost보다 F1이 높지는 않다. 다만 recall이 높고 구조가 가벼워, 운영 관점의 경량 시계열 비교 모델로 의미가 있다.

현재 API ensemble 비교:

| 비교 | F1 | Recall | Precision |
| --- | ---: | ---: | ---: |
| 기존 XGB + TS fallback | 0.1041 | 0.1273 | 0.0881 |
| 신규 XGB + TCN + TS | 0.1200 | 0.1636 | 0.0947 |

해석:

- TCN 단독 모델은 최종 우승 모델이라기보다 recall 중심의 경량 시계열 보조 모델이다.
- API ensemble에 TCN을 포함하면 기존 XGB+TS fallback보다 F1과 recall이 소폭 개선된다.
- 따라서 TCN의 핵심 가치는 "최종 대체 모델"보다 "LSTM과 Transformer 사이의 경량 비교 축"에 있다.

## 11. 기대 효과

- LSTM과 TS-Transformer 사이의 중간 비교 모델 확보
- 30일 시계열 데이터의 변화 패턴을 빠르게 탐지
- 기존 3D Tensor 데이터를 그대로 활용 가능
- 모델 학습 및 추론 속도 개선 가능
- 성능뿐 아니라 운영 효율성까지 비교 가능
- 프로젝트의 모델 실험 구조를 더 설득력 있게 확장

## 12. 재실행 명령

학습:

```bash
python -B scripts/train_tcn.py --epochs 50 --batch_size 512
```

평가:

```bash
python -B scripts/evaluate_pro.py
```

테스트:

```bash
python -B -m unittest tests.test_tcn_model
```

## 13. 한 줄 요약

TCN은 LSTM의 시계열 기억 한계를 보완하고, TS-Transformer보다 가벼운 방식으로 30일 행동 변화 패턴을 안정적으로 탐지하기 위한 경량 시계열 비교 모델이다.
