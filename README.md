# 🛰️ ChurnRadar: 고객 이탈 예측 및 자동 알림 시스템

> [cite_start]**2026 SW중심대학 기업연계 프로젝트 (11팀)** [cite: 1, 5]
> [cite_start]**개발 기간:** 2026. 03. 03 ~ 2026. 06. 19 [cite: 5]

## 📌 프로젝트 소개
[cite_start]고객 데이터를 분석하여 이탈 징후를 미리 탐지하고, 담당자에게 자동으로 알림을 발송하여 선제적 대응이 가능한 '골든타임'을 확보하는 자동화 파이프라인 시스템입니다. [cite: 39-43, 347]

## 👥 팀원 및 역할 분담
* [cite_start]**팀장 (데이터 준비):** Mendeley CRM 데이터 분석, 결측치 처리 및 SMOTE 클래스 불균형 보정 [cite: 274-279]
* [cite_start]**팀원 1 (AI 모델 + API):** ML 모델(LR, RF, XGBoost) 비교 실험 및 최적 모델 선정, FastAPI 서버 구현 [cite: 280-285]
* [cite_start]**팀원 2 (자동화 파이프라인):** n8n 스케줄러 설정, GPT 메시지 연동 및 Slack/Gmail 발송 자동화 [cite: 286-291]
* [cite_start]**팀원 3 (통합 및 최적화):** E2E 시스템 연결, API 응답 속도 개선 및 최종 데모 시나리오 구성 [cite: 292-297]

## 🛠️ 기술 스택
* [cite_start]**Data:** Mendeley Data (Bulgaria Telco CRM) [cite: 71, 78-80]
* [cite_start]**Machine Learning:** scikit-learn (Logistic Regression, Random Forest, XGBoost) [cite: 72, 131]
* [cite_start]**Backend:** FastAPI [cite: 73, 192]
* [cite_start]**Automation:** n8n [cite: 74, 221]
* [cite_start]**Communication:** Slack, Gmail, OpenAI GPT [cite: 75, 241, 247]

## 📁 프로젝트 구조
```text
ChurnRadar/
[cite_start]├── api/          # FastAPI 서버 및 API 스펙 정의 [cite: 285]
[cite_start]├── data/         # CRM 데이터셋 (원본 및 전처리 데이터) [cite: 275-276]
[cite_start]├── models/       # 학습 완료된 모델 파일 (.joblib) [cite: 284]
[cite_start]├── n8n/          # n8n 워크플로우 설정 파일 [cite: 287]
[cite_start]└── notebooks/    # EDA 및 AI 모델 실험용 Jupyter Notebook [cite: 282]
```

## 🚀 시작하기 (팀원 공통)
팀원들은 리포지토리를 클론한 후 아래 명령어를 통해 환경 설정을 완료해 주세요.

1. **가상환경 활성화 및 패키지 설치:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **작업 규칙:**
   * [cite_start]각자 맡은 역할의 폴더 내에서 작업을 진행합니다. [cite: 273-291]
   * 새로운 코드를 작성하기 전 반드시 `git pull`을 실행합니다.
   * [cite_start]공통으로 사용하는 `api` 폴더 수정 시 팀원들에게 공유합니다. [cite: 298]

## 🎯 목표 성능 지표
* [cite_start]**AI 모델:** F1-Score ≥ 0.80 달성 [cite: 131, 161, 325-327]
* [cite_start]**성능:** 감지부터 알림 도착까지 5초 이내 [cite: 330-334]
* [cite_start]**안정성:** 자동화 성공률 95% 이상 유지 [cite: 335-339]