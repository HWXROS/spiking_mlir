import torch
import torch.onnx
from torch.library import Library, impl

lib = Library("HcaSnn_ops", "DEF")

lib.define("HcaSnn_ifnode_op(Tensor v, float v_threshold, float v_reset) -> Tensor")
lib.define("HcaSnn_lifnode_op(Tensor v, float v_threshold, float v_reset, float tau, float decay_input) -> Tensor")
lib.define("HcaSnn_plifnode_op(Tensor v, Tensor tau, float v_threshold, float v_reset) -> Tensor")


@impl(lib, "HcaSnn_ifnode_op", "CPU")
def HcaSnn_ifnode_op_impl(v, v_threshold, v_reset):
    return v


@impl(lib, "HcaSnn_lifnode_op", "CPU")
def HcaSnn_lifnode_op_impl(v, v_threshold, v_reset, tau, decay_input):
    return v


@impl(lib, "HcaSnn_plifnode_op", "CPU")
def HcaSnn_plifnode_op_impl(v, tau, v_threshold, v_reset):
    return v


@impl(lib, "HcaSnn_ifnode_op", "Meta")
def HcaSnn_ifnode_op_meta(v, v_threshold, v_reset):
    return torch.empty_like(v)


@impl(lib, "HcaSnn_lifnode_op", "Meta")
def HcaSnn_lifnode_op_meta(v, v_threshold, v_reset, tau, decay_input):
    return torch.empty_like(v)


@impl(lib, "HcaSnn_plifnode_op", "Meta")
def HcaSnn_plifnode_op_meta(v, tau, v_threshold, v_reset):
    return torch.empty_like(v)


def HcaSnn_ifnode_op_symbolic(g, v, v_threshold, v_reset):
    if isinstance(v_threshold, torch._C.Value):
        v_threshold = v_threshold.node().t("value").item()
    if isinstance(v_reset, torch._C.Value):
        v_reset = v_reset.node().t("value").item()
    from torch.onnx.symbolic_helper import _get_tensor_sizes
    v_shape = _get_tensor_sizes(v)
    output = g.op("HcaSnn_ops::HcaSnn_ifnode_op", v, v_threshold_f=v_threshold, v_reset_f=v_reset)
    if v_shape is not None:
        output.setType(v.type().with_sizes(v_shape))
    return output


def HcaSnn_lifnode_op_symbolic(g, v, v_threshold, v_reset, tau, decay_input):
    if isinstance(v_threshold, torch._C.Value):
        v_threshold = v_threshold.node().t("value").item()
    if isinstance(v_reset, torch._C.Value):
        v_reset = v_reset.node().t("value").item()
    if isinstance(tau, torch._C.Value):
        tau = tau.node().t("value").item()
    if isinstance(decay_input, torch._C.Value):
        decay_input = decay_input.node().t("value").item()
    from torch.onnx.symbolic_helper import _get_tensor_sizes
    v_shape = _get_tensor_sizes(v)
    output = g.op("HcaSnn_ops::HcaSnn_lifnode_op", v, v_threshold_f=v_threshold, v_reset_f=v_reset, tau_f=tau, decay_input_f=decay_input)
    if v_shape is not None:
        output.setType(v.type().with_sizes(v_shape))
    return output


def HcaSnn_plifnode_op_symbolic(g, v, tau, v_threshold, v_reset):
    if isinstance(v_threshold, torch._C.Value):
        v_threshold = v_threshold.node().t("value").item()
    if isinstance(v_reset, torch._C.Value):
        v_reset = v_reset.node().t("value").item()
    if isinstance(tau, torch._C.Value):
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        v_shape = _get_tensor_sizes(v)
        output = g.op("HcaSnn_ops::HcaSnn_plifnode_op", v, tau, v_threshold_f=v_threshold, v_reset_f=v_reset)
        if v_shape is not None:
            output.setType(v.type().with_sizes(v_shape))
        return output
    else:
        tau_tensor = g.op("Constant", value_t=torch.tensor(tau))
        v_shape = _get_tensor_sizes(v)
        output = g.op("HcaSnn_ops::HcaSnn_plifnode_op", v, tau_tensor, v_threshold_f=v_threshold, v_reset_f=v_reset)
        if v_shape is not None:
            output.setType(v.type().with_sizes(v_shape))
        return output


torch.onnx.register_custom_op_symbolic("HcaSnn_ops::HcaSnn_ifnode_op", HcaSnn_ifnode_op_symbolic, 9)
torch.onnx.register_custom_op_symbolic("HcaSnn_ops::HcaSnn_lifnode_op", HcaSnn_lifnode_op_symbolic, 9)
torch.onnx.register_custom_op_symbolic("HcaSnn_ops::HcaSnn_plifnode_op", HcaSnn_plifnode_op_symbolic, 9)
