from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


def build_voting_ensemble(named_pipelines: dict) -> VotingClassifier:
    """학습된 Pipeline들로 Soft Voting 앙상블 구성.

    Args:
        named_pipelines: {"lr": Pipeline, "rf": Pipeline, ...}
    Returns:
        VotingClassifier (soft voting)
    """
    estimators = list(named_pipelines.items())
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
