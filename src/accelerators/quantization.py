import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
    float_qparams_weight_only_qconfig,
)
from typing import Set, Any, List

from utils.constants import DynamicQ, StaticQ, QAT


def quantization(
    model: torch.nn.Module,
    quantization_type: str,
    example_input: Any = torch.randn(4, 1, 4, 4),
    quantized_layers: List[torch.nn.Module] = {torch.nn.Linear},
):
    if quantization_type == DynamicQ:
        model_int8 = torch.quantization.quantize_dynamic(model, qconfig_spec=quantized_layers, dtype=torch.qint8)
    elif quantization_type == StaticQ:
        model.qconfig = torch.ao.quantization.default_qconfig
        model = model.to('cpu')
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
    else:
        raise NotImplementedError(f"{quantization_type} is not implemented.")

    return model_int8

