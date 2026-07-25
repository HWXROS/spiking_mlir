import copy
import torch
import torch.nn as nn
from .register import *
from spikingjelly.activation_based import layer, neuron
from spikingjelly.clock_driven import neuron as cd_neuron


# ============================================================================
# BatchNorm Auto-Fusion (model-agnostic)
# ============================================================================

def _fuse_conv_bn(conv, bn):
    """Fuse ConvNd + BatchNormNd -> ConvNd(bias=True)."""
    assert conv.out_channels == bn.num_features, \
        f"Channel mismatch: conv={conv.out_channels}, bn={bn.num_features}"
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = gamma / std

    if isinstance(conv, nn.Conv1d):
        fused_weight = conv.weight * scale.view(-1, 1, 1)
        ConvCls = nn.Conv1d
    elif isinstance(conv, nn.Conv2d):
        fused_weight = conv.weight * scale.view(-1, 1, 1, 1)
        ConvCls = nn.Conv2d
    elif isinstance(conv, nn.Conv3d):
        fused_weight = conv.weight * scale.view(-1, 1, 1, 1, 1)
        ConvCls = nn.Conv3d
    else:
        raise TypeError(f"Unsupported conv type: {type(conv)}")

    if conv.bias is not None:
        fused_bias = (conv.bias - mean) * scale + beta
    else:
        fused_bias = beta - mean * scale

    fused = ConvCls(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=True
    )
    fused.weight.data = fused_weight
    fused.bias.data = fused_bias
    return fused


def _fuse_linear_bn(linear, bn):
    """Fuse Linear + BatchNorm1d -> Linear(bias=True)."""
    assert linear.out_features == bn.num_features, \
        f"Feature mismatch: linear={linear.out_features}, bn={bn.num_features}"
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = gamma / std
    fused_weight = linear.weight * scale.view(-1, 1)
    if linear.bias is not None:
        fused_bias = (linear.bias - mean) * scale + beta
    else:
        fused_bias = beta - mean * scale
    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)
    fused.weight.data = fused_weight
    fused.bias.data = fused_bias
    return fused


def _try_fuse_pair(parent_module, child_name):
    """Check if child is Conv/Linear followed by BN; if so, fuse and replace BN with Identity."""
    children = list(parent_module.named_children())
    names = [name for name, _ in children]
    modules = [mod for _, mod in children]
    try:
        idx = names.index(child_name)
    except ValueError:
        return False
    if idx + 1 >= len(modules):
        return False

    current = modules[idx]
    next_mod = modules[idx + 1]
    next_name = names[idx + 1]

    fused = None
    if isinstance(current, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and \
       isinstance(next_mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        fused = _fuse_conv_bn(current, next_mod)
    elif isinstance(current, nn.Linear) and isinstance(next_mod, nn.BatchNorm1d):
        fused = _fuse_linear_bn(current, next_mod)
    else:
        return False

    setattr(parent_module, child_name, fused)
    setattr(parent_module, next_name, nn.Identity())
    return True


def auto_fuse_bn(model, inplace=False, verbose=False):
    """
    Recursively scan model and fuse all Conv+BN / Linear+BN pairs.

    Args:
        model: PyTorch nn.Module.
        inplace: If True, modify model in-place.
        verbose: Print fusion progress.

    Returns:
        The fused model.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()

    fusion_count = 0
    def _recursive_scan(module, path=""):
        nonlocal fusion_count
        for name, child in list(module.named_children()):
            full_name = f"{path}.{name}" if path else name
            if _try_fuse_pair(module, name):
                fusion_count += 1
                if verbose:
                    print(f"  [Fused] {full_name} + BN")
            _recursive_scan(child, full_name)

    _recursive_scan(model)
    if verbose:
        print(f"Total fused pairs: {fusion_count}")
    return model


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
    # clock_driven neurons (MultiStepLIFNode, MultiStepParametricLIFNode, etc.)
    # These inherit from clock_driven.LIFNode/ParametricLIFNode and share attributes
    # with their activation_based counterparts.
    elif isinstance(original_layer, cd_neuron.LIFNode):
        v_reset = getattr(original_layer, 'v_reset', 0.0)
        v_reset = v_reset if v_reset is not None else float('nan')
        return LIFNodeWrapper(
            v_threshold=original_layer.v_threshold,
            v_reset=v_reset,
            tau=getattr(original_layer, 'tau', 2.0),
            decay_input=float(getattr(original_layer, 'decay_input', True))
        )
    elif isinstance(original_layer, cd_neuron.ParametricLIFNode):
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
            neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode,
            cd_neuron.LIFNode, cd_neuron.ParametricLIFNode,
        )):
            new_layer = replace_layer_with_nn(child)
            setattr(model, name, new_layer)
        elif len(list(child.children())) > 0:
            replace_layers_recursive(child)
    return model


def import_onnx_model(model: nn.Module, input_shape: tuple, output_path: str = "model.onnx", mlir_output_path: str = None, core_layer=None, fuse_bn: bool = True):
    if fuse_bn:
        model = auto_fuse_bn(model, inplace=False, verbose=True)
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
