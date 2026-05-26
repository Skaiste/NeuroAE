from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - import guard for environments without xgboost
    XGBClassifier = None


def build_logreg(config, num_classes):
    return LogisticRegression(
        max_iter=int(config.get("max_iter", 1000)),
        C=float(config.get("C", 1.0)),
        solver=str(config.get("solver", "lbfgs")),
        class_weight=config.get("class_weight", "balanced"),
    )


def build_svm(config):
    return SVC(
        C=float(config.get("C", 1.0)),
        kernel=str(config.get("kernel", "rbf")),
        gamma=config.get("gamma", "scale"),
        probability=bool(config.get("probability", True)),
        class_weight=config.get("class_weight", "balanced"),
    )


def build_xgboost(config, num_classes):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed but model.name='xgboost' was requested.")
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    return XGBClassifier(
        n_estimators=int(config.get("n_estimators", 200)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 1.0)),
        colsample_bytree=float(config.get("colsample_bytree", 1.0)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        objective=objective,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        num_class=num_classes if num_classes > 2 else None,
        random_state=int(config.get("random_state", 42)),
    )
