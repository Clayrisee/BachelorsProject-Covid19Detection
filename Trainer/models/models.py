from models.ensemble_effiecientnet_vit import EnsembleEfficientNetViT
from models.ensemble_vit_convnext import EnsembleConvNextVIT

def create_model(model_name:str, **kwargs):
    """
    Create model based on model_name.
    Args:
        model_name (str): model name
    Return:
        model
    """
    if model_name == 'ensemble_convnext_vit':
        model = EnsembleConvNextVIT(**kwargs)
    elif model_name == 'ensemble_efficientnet_vit':
        model = EnsembleEfficientNetViT(**kwargs)
    else:
        raise NotImplementedError
    return model