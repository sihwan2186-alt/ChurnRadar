# 🛰️ ChurnRadar: 고객 이탈 예측 및 자동 알림 시스템

> **2026 SW중심대학 기업연계 프로젝트 (11팀)** 
> **개발 기간:** 2026. 03. 03 ~ 2026. 06. 19 

## 📌 프로젝트 소개
고객 데이터를 분석하여 이탈 징후를 미리 탐지하고, 담당자에게 자동으로 알림을 발송하여 선제적 대응이 가능한 '골든타임'을 확보하는 자동화 파이프라인 시스템입니다.

## 👥 팀원 및 역할 분담
* **팀장 (데이터 준비):** Mendeley CRM 데이터 분석, 결측치 처리 및 SMOTE 클래스 불균형 보정
* **팀원 1 (AI 모델 + API):** ML 모델(LR, RF, XGBoost) 비교 실험 및 최적 모델 선정, FastAPI 서버 구현 
* **팀원 2 (자동화 파이프라인):** n8n 스케줄러 설정, GPT 메시지 연동 및 Slack/Gmail 발송 자동화
* **팀원 3 (통합 및 최적화):** E2E 시스템 연결, API 응답 속도 개선 및 최종 데모 시나리오 구성

## 🛠️ 기술 스택
* **Data:** Mendeley Data (Bulgaria Telco CRM) 
* **Machine Learning:** scikit-learn (Logistic Regression, Random Forest, XGBoost) 
* **Backend:** FastAPI 
* **Automation:** n8n 
* **Communication:** Slack, Gmail, OpenAI GPT 

## 📁 프로젝트 구조
```text
ChurnRadar/
├── api/          # FastAPI 서버 및 API 스펙 정의 
├── data/         # CRM 데이터셋 (원본 및 전처리 데이터) 
├── models/       # 학습 완료된 모델 파일 (.joblib) 
├── n8n/          # n8n 워크플로우 설정 파일 
└── notebooks/    # EDA 및 AI 모델 실험용 Jupyter Notebook
```

## 🚀 시작하기 (팀원 공통)
팀원들은 리포지토리를 클론한 후 아래 명령어를 통해 환경 설정을 완료해 주세요.

1. **가상환경 활성화 및 패키지 설치:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **작업 규칙:**
   * 각자 맡은 역할의 폴더 내에서 작업을 진행합니다. 
   * 새로운 코드를 작성하기 전 반드시 `git pull`을 실행합니다.
   * 공통으로 사용하는 `api` 폴더 수정 시 팀원들에게 공유합니다. 

## 🎯 목표 성능 지표
* **AI 모델:** F1-Score ≥ 0.80 달성 
* **성능:** 감지부터 알림 도착까지 5초 이내 
* **안정성:** 자동화 성공률 95% 이상 유지


