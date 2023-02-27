from accelerators.accelerator_factory import AcceleratorFactory
from utils.constants import TRANSFORMER_XL


def apply_accelerator(model_name: str, model, accelerator_type: str = None, **accelerator_args):
    """
    Apply an accelerator to a pretrained model
    :param model_name: Name of the model
    :param model: Pretrained model loaded from HuggingFace
    :param accelerator_type: Name of the accelerator technique to use
    :param accelerator_args: Optional arguments for the accelerator
    """
    accelerator = AcceleratorFactory().get_accelerator(accelerator_type)

    for i, layer in enumerate(model.transformer.layers):
        if model_name == TRANSFORMER_XL:
            model.transformer.layers[i].pos_ff.CoreNet[0] = accelerator(model.transformer.layers[i].pos_ff.CoreNet[0],
                                                                        **accelerator_args)
            model.transformer.layers[i].pos_ff.CoreNet[3] = accelerator(model.transformer.layers[i].pos_ff.CoreNet[3],
                                                                        **accelerator_args)
