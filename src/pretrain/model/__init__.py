from pretrain.config.schema import ModelConfig
from pretrain.model.llama3 import Llama3, build_model as _build_llama3
from pretrain.model.precision import MixedPrecisionPolicyFactory

# Importing modules registers them with the swap registries.
from pretrain.model.modules import norm_repop as _norm_repop  # noqa: F401
from pretrain.model.modules import attention_repop as _attention_repop  # noqa: F401
from pretrain.model.modules import embedding_repop as _embedding_repop  # noqa: F401
from pretrain.model.modules import ffn_repop as _ffn_repop  # noqa: F401
from pretrain.model.modules import rope as _rope  # noqa: F401


def build_model(cfg: ModelConfig, device=None):
    return _build_llama3(cfg, device=device)


__all__ = [
    "Llama3",
    "build_model",
    "MixedPrecisionPolicyFactory",
]
