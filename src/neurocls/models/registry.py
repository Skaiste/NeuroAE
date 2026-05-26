from __future__ import annotations

from .braingnn import BrainGNNClassifier
from .classical import build_logreg, build_svm, build_xgboost
from .mlp import ShallowMLP


def create_model(model_config, input_shape, num_classes):
    cfg = dict(model_config.get("model", {}))
    name = str(cfg.get("name", "logreg")).lower()
    if name == "logreg":
        return {"name": name, "family": "sklearn", "model": build_logreg(cfg, num_classes)}
    if name == "svm":
        return {"name": name, "family": "sklearn", "model": build_svm(cfg)}
    if name == "xgboost":
        return {"name": name, "family": "sklearn", "model": build_xgboost(cfg, num_classes)}
    if name == "mlp":
        input_dim = int(input_shape[-1])
        return {
            "name": name,
            "family": "torch_vector",
            "model": ShallowMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=cfg.get("hidden_dims", [256, 64]),
                dropout=cfg.get("dropout", 0.2),
            ),
        }
    if name == "braingnn":
        node_feature_dim = int(input_shape[-1])
        return {
            "name": name,
            "family": "torch_graph",
            "model": BrainGNNClassifier(
                node_feature_dim=node_feature_dim,
                num_classes=num_classes,
                hidden_dims=cfg.get("hidden_dims", [128, 64]),
                pool_ratios=cfg.get("pool_ratios", [0.5, 0.5]),
                dropout=cfg.get("dropout", 0.2),
                aux_loss_weight=cfg.get("aux_loss_weight", 0.1),
            ),
        }
    raise ValueError(f"Unsupported model.name {cfg.get('name')!r}.")

