from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def scale_features() -> Pipeline:
    """
    수치형 변수에 대한 전처리 파이프라인을 생성하여 반환합니다.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return num_pipe
