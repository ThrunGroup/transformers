import os
import torch
from typing import List

from accelerators.accelerator_factory import AcceleratorFactory
from accelerators.quantization import collect_stats, compute_amax, quantization
from utils.constants import TRANSFORMER_XL, GPT2, GPT2_LARGE, OPT, OPT_350M, QUANTIZATION, SVD, PRUNING, DynamicQ
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

def apply_accelerator(model_name: str,
                      model,
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

    if layers_to_accelerate is None:
        layers_to_accelerate = list(range(100))  # Hard-coded

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

    elif OPT in model_name:
        use_cuda = accelerator_args["use_cuda"]
        if accelerator_type == QUANTIZATION:
            quantization(model, DynamicQ)
            # example_input = accelerator_args["example_input"]
            # quant_desc_input = QuantDescriptor(calib_method='histogram')
            # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
            # for i, block in enumerate(model.model.decoder.layers):
            #     block.fc1 = accelerator(block.fc1, use_cuda=use_cuda)
            #     block.fc2 = accelerator(block.fc2, use_cuda=use_cuda)
            # with torch.no_grad():
            #     collect_stats(model, data_loader=[example_input], num_batches=1)
            #     compute_amax(model, method="percentile", percentile=99.99)
            # if use_cuda:
            #     model.cuda()
        elif accelerator_type in [SVD, PRUNING]:
            k = accelerator_args["k"]
            for i, block in enumerate(model.model.decoder.layers):
                block.fc1 = accelerator(block.fc1, k=k, is_conv1d=False)
                block.fc2 = accelerator(block.fc2, k=k, is_conv1d=False)



    return model, accelerated_layers