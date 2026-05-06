import argparse
import pandas as pd
import logging
from pathlib import Path
import sys

# 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api.model_handler import predict_churn

def main():
    parser = argparse.ArgumentParser(description="ChurnRadar B2B 배치 추론 파이프라인")
    parser.add_argument("--input", required=True, help="입력 CSV 경로")
    parser.add_argument("--output", required=True, help="결과 CSV 저장 경로")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return

    logger.info(f"데이터 로드 중... ({input_path})")
    df = pd.read_csv(input_path)
    total_customers = len(df)
    logger.info(f"총 {total_customers}명의 고객 데이터 확인됨.")

    results = []
    
    # 500건마다 진행상황 출력을 위해 iterrows 활용
    for i, row in df.iterrows():
        if i > 0 and i % 500 == 0:
            logger.info(f"추론 진행 중... {i}/{total_customers}")
            
        data_dict = {
            "customer_id": str(row.get("PID", f"CUST_{i}")),
            "total_subs": int(row.get("Total_SUBs", 1)),
            "avg_mobile_revenue": float(row.get("AvgMobileRevenue", 0)),
            "avg_fix_revenue": float(row.get("AvgFIXRevenue", 0)),
            "total_revenue": float(row.get("TotalRevenue", 0)),
            "arpu": float(row.get("ARPU", 0)),
            "active_subscribers": int(row.get("Active_subscribers", 0)),
            "not_active_subscribers": float(row.get("Not_Active_subscribers", 0)),
            "crm_segment": str(row.get("CRM_PID_Value_Segment", "Unknown")),
            "effective_segment": str(row.get("EffectiveSegment", "Unknown")),
            "history_arpu": None
        }
        
        ka_name = str(row.get("KA_name", "Unknown"))
        
        pred = predict_churn(data_dict)
        
        # 결과 합치기
        result_row = {
            "PID": data_dict["customer_id"],
            "KA_name": ka_name,
            "ARPU": data_dict["arpu"],
            "xgb_probability": pred["xgb_probability"],
            "ts_probability": pred["ts_probability"],
            "churn_probability": pred["churn_probability"],
            "churn_prediction": pred["churn_prediction"],
            "risk_level": pred["risk_level"],
            "expected_revenue_loss": pred["expected_revenue_loss"]
        }
        results.append(result_row)
        
    result_df = pd.DataFrame(results)
    
    # KA_name 기반 B2B 그룹 위험도 계산 (Contagion Network)
    logger.info("KA_name 기반 그룹 위험도(Contagion Network) 분석 시작...")
    
    group_stats = result_df.groupby("KA_name").agg(
        group_avg_prob=("churn_probability", "mean"),
        group_max_prob=("churn_probability", "max"),
        group_size=("PID", "count")
    ).reset_index()
    
    # 그룹 내 한 명이라도 0.70 이상이면 알림 대상
    group_stats["group_alert"] = group_stats["group_max_prob"] >= 0.70
    
    # 원본 결과와 병합
    final_df = pd.merge(result_df, group_stats, on="KA_name", how="left")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    # 요약 출력
    total_churners = final_df["churn_prediction"].sum()
    high_risk_count = (final_df["risk_level"] == "HIGH").sum()
    total_loss = final_df["expected_revenue_loss"].sum()
    group_alerts = group_stats["group_alert"].sum()
    
    logger.info("=== B2B 배치 추론 요약 ===")
    logger.info(f"전체 고객 수: {total_customers}명")
    logger.info(f"이탈 예측 수: {total_churners}명")
    logger.info(f"HIGH 위험 고객 수: {high_risk_count}명")
    logger.info(f"그룹 경보 발령(담당자 수): {group_alerts}건")
    logger.info(f"예상 총 매출 손실액: {total_loss:.2f} BGN")
    logger.info(f"결과 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
