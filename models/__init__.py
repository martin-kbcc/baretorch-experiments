from models.transformer import CausalTransformerLM
from models.gla import CausalGLALM
from models.gdn2 import CausalGDN2LM
from models.mamba3 import CausalMamba3LM
from models.cs_lrad import CausalLRADLM
from models.ccrs import CausalCCRSLM
from models.ckts import CausalCKTSLM
from models.cbkc import CausalCBKCLM
from models.cofe import CausalCOFELM
from models.cs_hde import CausalHDELM
from models.cs_dme import CausalDMELM
from models.cs_apex import CausalAPEXLM
from models.cs_nexus import CausalNEXUSLM
from models.cs_cann import CausalCANNLM
from models.cs_delta_nexus import CausalDeltaNexusLM
from models.hdsa import CausalHDSALM
from models.pg_rope import CausalPGRoPELM
from models.rgda import CausalRGDALM

MODEL_REGISTRY = {
    "transformer":    CausalTransformerLM,
    "gla":            CausalGLALM,
    "gdn2":           CausalGDN2LM,
    "mamba3":         CausalMamba3LM,
    "cs_lrad":        CausalLRADLM,
    "ccrs":           CausalCCRSLM,
    "ckts":           CausalCKTSLM,
    "cbkc":           CausalCBKCLM,  
    "cofe":           CausalCOFELM,
    "cs_hde":         CausalHDELM,
    "cs_dme":         CausalDMELM,
    "cs_apex":        CausalAPEXLM,
    "cs_nexus":       CausalNEXUSLM,
    "cs_cann":        CausalCANNLM,
    "cs_delta_nexus": CausalDeltaNexusLM,
    "hdsa":           CausalHDSALM,
    "pg_rope":        CausalPGRoPELM,
    "rgda":           CausalRGDALM,
}

def make_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Architecture '{name}' is not registered in BareTorch model catalog.")
    model_class = MODEL_REGISTRY[name]
    if model_class is None:
        raise ImportError(f"Architecture file found for '{name}', but code contains compilation errors.")
    return model_class(**kwargs)