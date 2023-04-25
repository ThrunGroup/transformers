import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
    float_qparams_weight_only_qconfig,
)
from tqdm import tqdm
from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from typing import Set, Any, List

from utils.constants import DynamicQ, StaticQ, QAT, QUANTIZATION_NVIDIA


def quantization(
    model: torch.nn.Module,
    quantization_type: str,
    example_input: Any = torch.randn(4, 1, 4, 4),
    quantized_layers: List[torch.nn.Module] = {torch.nn.Linear},
    model_name: str = "",
):
    if quantization_type == DynamicQ:
        model_int8 = torch.quantization.quantize_dynamic(
            model, qconfig_spec=quantized_layers, dtype=torch.qint8, inplace=True
        )

    elif quantization_type == StaticQ:
        model.qconfig = torch.ao.quantization.default_qconfig
        model = model.to("cpu")
        model.eval()
        backend = "fbgemm"
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend

        model.model.decoder.embed_tokens.qconfig = float_qparams_weight_only_qconfig
        # model.model.decoder = torch.ao.quantization.QuantWrapper(model.model.decoder)
        torch.quantization.prepare(model.model.decoder, inplace=True)
        torch.quantization.convert(model.model.decoder, inplace=True)
        model(example_input)
        model_int8 = model

        # model_fp32_fused = torch.ao.quantization.fuse_modules(model, {torch.nn.Linear, torch.nn.ReLU})
        # model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        # model_fp32_prepared(example_input)
        # model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        # qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        # model.eval()
        # extra_positional_args = [None] * 8
        # example_input = [example_input] + extra_positional_args
        # model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, tuple(example_input))
        # model_int8 = quantize_fx.convert_fx(model_prepared)
    elif quantization_type == QUANTIZATION_NVIDIA:
        quant_desc_input = QuantDescriptor(calib_method="histogram")
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        model_int8 = model

        with torch.no_grad():
            collect_stats(model_int8, data_loader=[example_input], num_batches=1)
            compute_amax(model_int8, method="percentile", percentile=99.99)
        # model = ORTModelForCausalLM(model_name)
        # quantizer = ORTQuantizer.from_pretrained(model)
        # dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        # model_int8 = quantizer.quantize(quantization_config=dqconfig)

    else:
        raise NotImplementedError(f"{quantization_type} is not implemented.")

    return model_int8


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(name, module._calibrator)
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, sample in enumerate(data_loader):
        model(sample[0])
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
