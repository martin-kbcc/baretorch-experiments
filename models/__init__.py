from models.transformer import CausalTransformerLM
from models.gla import CausalGLALM
from models.gdn2 import CausalGDN2LM
from models.mamba3 import CausalMamba3LM
from models.cs_lrad import CausalLRADLM
from models.ccrs import CausalCCRSLM
from models.ckts import CausalCKTSLM
from models.cbkc import CausalCBKCLM
from models.cofe import CausalCOFELM

MODEL_REGISTRY = {
    "transformer": CausalTransformerLM,
    "gla":         CausalGLALM,
    "gdn2":        CausalGDN2LM,
    "mamba3":      CausalMamba3LM,
    "cs_lrad":     CausalLRADLM,
    "ccrs":        CausalCCRSLM,
    "ckts":        CausalCKTSLM,
    "cbkc":        CausalCBKCLM,  # Inherently sub-quadratic Block-Kronecker Cascade model track
    "cofe":        CausalCOFELM   # Causal Orthogonal Feedback Engine tracking delta-net behaviors
}

def make_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Architecture '{name}' is not registered in BareTorch model catalog.")
    model_class = MODEL_REGISTRY[name]
    if model_class is None:
        raise ImportError(f"Architecture file found for '{name}', but code contains compilation errors.")
    return model_class(**kwargs)