import os
import tqdm

from accelerators.accelerator_factory import AcceleratorFactory
from utils.constants import TRANSFORMER_XL, GPT2, GPT2_LARGE


def apply_accelerator(model_name: str, model, accelerator_type: str = None, **accelerator_args):
    """
    Apply an accelerator to a pretrained model
    :param model_name: Name of the model
    :param model: Pretrained model loaded from HuggingFace
    :param accelerator_type: Name of the accelerator technique to use
    :param accelerator_args: Optional arguments for the accelerator
    """
    if accelerator_type is None:
        return
    accelerator = AcceleratorFactory().get_accelerator(accelerator_type)
    if model_name == TRANSFORMER_XL:
        for i, layer in enumerate(model.transformer.layers):
            model.transformer.layers[i].pos_ff.CoreNet[0] = accelerator(
                model.transformer.layers[i].pos_ff.CoreNet[0], **accelerator_args
            )
            model.transformer.layers[i].pos_ff.CoreNet[3] = accelerator(
                model.transformer.layers[i].pos_ff.CoreNet[3], **accelerator_args
            )
    elif GPT2 in model_name:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), model_name)
        k = accelerator_args["k"]
        for i, decoder_block in enumerate(model.transformer.h):
            checkpoint_path = os.path.join(checkpoint_dir, f"layer_{i}")
            decoder_block.mlp.c_fc = accelerator(
                decoder_block.mlp.c_fc, k=k, checkpoint_path=checkpoint_path + "_c_fc", is_conv1d=True
            )
            decoder_block.mlp.c_proj = accelerator(
                decoder_block.mlp.c_proj, k=k, checkpoint_path=checkpoint_path + "_c_proj", is_conv1d=True
            )
        # checkpoint_path = os.path.join(checkpoint_dir, f"lm_head")
        # model.lm_head = accelerator(model.lm_head, k=k, checkpoint_path=checkpoint_path, is_conv1d=False)


