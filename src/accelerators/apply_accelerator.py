import os
from typing import List

from accelerators.accelerator_factory import AcceleratorFactory
from utils.constants import TRANSFORMER_XL, GPT2, GPT2_LARGE
from utils.parse_string import parse_string


def apply_accelerator(model_name: str, model,
                      layers_to_accelerate: List[int] = None,
                      accelerator_type: str = None,
                      **accelerator_args):
    """
    Apply an accelerator to a pretrained model

    :param model_name: Name of the model
    :param model: Pretrained model loaded from HuggingFace
    :param layers_to_accelerate: Which layers to accelerate
    :param accelerator_type: Name of the accelerator technique to use
    :param accelerator_args: Optional arguments for the accelerator
    :return List of names of the accelerated layers
    """
    if accelerator_type is None:
        return

    accelerated_layers = []

    accelerator = AcceleratorFactory().get_accelerator(accelerator_type)
    if model_name == TRANSFORMER_XL:
        for i, layer in enumerate(model.transformer.layers):
            if i in layers_to_accelerate:
                model.transformer.layers[i].pos_ff.CoreNet[0] = accelerator(
                    layer.pos_ff.CoreNet[0], **accelerator_args
                )
                model.transformer.layers[i].pos_ff.CoreNet[3] = accelerator(
                    layer.pos_ff.CoreNet[3], **accelerator_args
                )
                # TODO: Update the layer name properly
                accelerated_layers.extend([f"{i}.pos_ff.CoreNet", f"{i}.mlp.c_proj"])
    elif GPT2 in model_name:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), model_name)

        k = accelerator_args["k"]
        for i, block in enumerate(model.transformer.h):
            if i in layers_to_accelerate:
                # Only accelerate the specified layers
                checkpoint_path = os.path.join(checkpoint_dir, f"layer_{i}")
                block.mlp.c_fc = accelerator(
                    block.mlp.c_fc, k=k, checkpoint_path=checkpoint_path + "_c_fc", is_conv1d=True
                )
                block.mlp.c_proj = accelerator(
                    block.mlp.c_proj, k=k, checkpoint_path=checkpoint_path + "_c_proj", is_conv1d=True
                )
                accelerated_layers.extend([f"transformer.h.{i}.mlp.c_fc", f"transformer.h.{i}.mlp.c_proj"])
        # checkpoint_path = os.path.join(checkpoint_dir, f"lm_head")
        # model.lm_head = accelerator(model.lm_head, k=k, checkpoint_path=checkpoint_path, is_conv1d=False)

    return accelerated_layers