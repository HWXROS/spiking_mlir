import torch
import torch.nn as nn
from .register import *
from spikingjelly.activation_based import layer, neuron


class IFNodeWrapper(nn.Module):
    def __init__(self, v_threshold: float, v_reset: float):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.HcaSnn_ops.HcaSnn_ifnode_op(x, self.v_threshold, self.v_reset)


class LIFNodeWrapper(nn.Module):
    def __init__(self, v_threshold: float, v_reset: float, tau: float, decay_input: float):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau = tau
        self.decay_input = decay_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.HcaSnn_ops.HcaSnn_lifnode_op(x, self.v_threshold, self.v_reset, self.tau, self.decay_input)


class PLIFNodeWrapper(nn.Module):
    def __init__(self, w: torch.Tensor, v_threshold: float, v_reset: float):
        super().__init__()
        self.w = w
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.HcaSnn_ops.HcaSnn_plifnode_op(x, self.w, self.v_threshold, self.v_reset)


class SNNExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, core_layer=None):
        super().__init__()
        self.model = model
        self.core_layers = None
        
        if core_layer is not None:
            if isinstance(core_layer, (list, tuple)):
                self.core_layers = nn.ModuleList(core_layer)
            elif isinstance(core_layer, nn.Module):
                self.core_layers = nn.ModuleList([core_layer])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 4:
            x = x.flatten(0, -5)
        if self.core_layers is not None:
            for layer in self.core_layers:
                x = layer(x)
            return x
        else:
            return self.model(x)


def copy_weights(source, target):
    with torch.no_grad():
        if hasattr(source, 'weight') and source.weight is not None:
            target.weight.copy_(source.weight.data)
        if hasattr(source, 'bias') and source.bias is not None:
            target.bias.copy_(source.bias.data)
        if (hasattr(source, 'running_mean') and source.running_mean is not None and
            hasattr(target, 'running_mean') and target.running_mean is not None):
            target.running_mean.copy_(source.running_mean)
            target.running_var.copy_(source.running_var)
            if hasattr(source, 'num_batches_tracked'):
                target.num_batches_tracked.copy_(source.num_batches_tracked)


def replace_layer_with_nn(original_layer):
    if isinstance(original_layer, layer.Flatten):
        return nn.Flatten(
            start_dim=original_layer.start_dim,
            end_dim=original_layer.end_dim
        )
    elif isinstance(original_layer, layer.Linear):
        new_layer = nn.Linear(
            original_layer.in_features,
            original_layer.out_features,
            bias=original_layer.bias is not None
        )
        copy_weights(original_layer, new_layer)
        return new_layer
    elif isinstance(original_layer, layer.Conv2d):
        new_layer = nn.Conv2d(
            original_layer.in_channels,
            original_layer.out_channels,
            original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            groups=original_layer.groups,
            bias=original_layer.bias is not None,
            padding_mode=original_layer.padding_mode
        )
        copy_weights(original_layer, new_layer)
        return new_layer
    elif isinstance(original_layer, layer.Conv1d):
        new_layer = nn.Conv1d(
            original_layer.in_channels,
            original_layer.out_channels,
            original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            groups=original_layer.groups,
            bias=original_layer.bias is not None,
            padding_mode=original_layer.padding_mode
        )
        copy_weights(original_layer, new_layer)
        return new_layer
    elif isinstance(original_layer, layer.MaxPool2d):
        return nn.MaxPool2d(
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            return_indices=original_layer.return_indices,
            ceil_mode=original_layer.ceil_mode
        )
    elif isinstance(original_layer, layer.AvgPool2d):
        return nn.AvgPool2d(
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            ceil_mode=original_layer.ceil_mode,
            count_include_pad=original_layer.count_include_pad,
            divisor_override=original_layer.divisor_override
        )
    elif isinstance(original_layer, layer.BatchNorm2d):
        new_layer = nn.BatchNorm2d(
            num_features=original_layer.num_features,
            eps=original_layer.eps,
            momentum=original_layer.momentum,
            affine=original_layer.affine,
            track_running_stats=True
        )
        copy_weights(original_layer, new_layer)
        nn.init.zeros_(new_layer.running_mean)
        nn.init.ones_(new_layer.running_var)
        return new_layer
    elif isinstance(original_layer, layer.Dropout):
        return nn.Dropout(
            p=original_layer.p,
            inplace=original_layer.inplace
        )
    elif isinstance(original_layer, neuron.IFNode):
        v_reset = getattr(original_layer, 'v_reset', 0.0)
        v_reset = v_reset if v_reset is not None else float('nan')
        return IFNodeWrapper(
            v_threshold=original_layer.v_threshold,
            v_reset=v_reset
        )
    elif isinstance(original_layer, neuron.LIFNode):
        v_reset = getattr(original_layer, 'v_reset', 0.0)
        v_reset = v_reset if v_reset is not None else float('nan')
        return LIFNodeWrapper(
            v_threshold=original_layer.v_threshold,
            v_reset=v_reset,
            tau=getattr(original_layer, 'tau', 2.0),
            decay_input=float(getattr(original_layer, 'decay_input', True))
        )
    elif isinstance(original_layer, neuron.ParametricLIFNode):
        w = original_layer.w.data if hasattr(original_layer, 'w') and original_layer.w is not None else torch.tensor(0.0)
        v_reset = getattr(original_layer, 'v_reset', 0.0)
        v_reset = v_reset if v_reset is not None else float('nan')
        return PLIFNodeWrapper(
            w=w,
            v_threshold=original_layer.v_threshold,
            v_reset=v_reset
        )
    else:
        return original_layer


def replace_layers_recursive(model: nn.Module) -> nn.Module:
    for name, child in model.named_children():
        if isinstance(child, (
            layer.Flatten, layer.Linear, layer.Conv2d, layer.Conv1d,
            layer.MaxPool2d, layer.AvgPool2d, layer.BatchNorm2d, layer.Dropout,
            neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode
        )):
            new_layer = replace_layer_with_nn(child)
            setattr(model, name, new_layer)
        elif len(list(child.children())) > 0:
            replace_layers_recursive(child)
    return model


def import_onnx_model(model: nn.Module, input_shape: tuple, output_path: str = "model.onnx", mlir_output_path: str = None, core_layer=None):
    model = replace_layers_recursive(model)
    model.eval()
    
    if core_layer is not None:
        if isinstance(core_layer, (list, tuple)):
            for i, layer in enumerate(core_layer):
                if not isinstance(layer, nn.Module):
                    raise ValueError(f"core_layer[{i}] 不是 nn.Module 类型，请检查传入的层是否正确")
        elif not isinstance(core_layer, nn.Module):
            raise ValueError("core_layer 必须是 nn.Module 类型或 nn.Module 列表，请检查传入的层是否正确")
    
    wrapped_model = SNNExportWrapper(model, core_layer)
    
    dummy_input = torch.randn(*input_shape)
    
    try:
        wrapped_model(dummy_input)
    except Exception as e:
        raise RuntimeError(
            f"模型前向传播失败，可能原因：\n"
            f"  1. input_shape 与模型输入不匹配\n"
            f"  2. core_layer 选择错误（当前: {core_layer}）\n"
            f"  3. 模型结构不支持当前导出方式\n"
            f"原始错误: {e}"
        ) from e
    
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamo=False
        )
    except Exception as e:
        raise RuntimeError(
            f"ONNX 导出失败，可能原因：\n"
            f"  1. core_layer 未正确指定，导致导出了包含时间维度操作的完整 forward\n"
            f"  2. 模型包含不支持导出的操作\n"
            f"建议：尝试指定 core_layer 参数，例如 core_layer=model.conv_fc\n"
            f"原始错误: {e}"
        ) from e
    
    print(f"Export completed! Saved to {output_path}")

    if mlir_output_path is not None:
        import onnx
        from onnx import shape_inference
        from torch_mlir.extras import onnx_importer
        from torch_mlir.dialects import torch as torch_d
        from torch_mlir.ir import Context

        model_proto = onnx.load(output_path)
        model_proto = shape_inference.infer_shapes(model_proto)
        context = Context()
        torch_d.register_dialect(context)
        config = onnx_importer.Config()
        model_info = onnx_importer.ModelInfo(model_proto, config=config)
        m = model_info.create_module(context=context).operation
        imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m)
        imp.import_all()
        with open(mlir_output_path, "wt") as f:
            print(m.get_asm(), file=f)
        print(f"MLIR export completed! Saved to {mlir_output_path}")

    return model
