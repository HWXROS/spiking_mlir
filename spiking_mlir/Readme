# Spiking MLIR

将 SpikingJelly 脉冲神经网络模型导出为 ONNX 和 MLIR 格式。

## 工作流程

```
SpikingJelly 模型 (layer.Linear, neuron.LIFNode)
           ↓  trans.py (层替换)
标准 PyTorch 模型 (nn.Linear, nn.Flatten, 自定义算子)
           ↓  register.py (ONNX 算子注册)
           ↓  torch.onnx.export
        ONNX 模型 (包含 HcaSnn_ifnode_op 等自定义算子)
           ↓  onnx.shape_inference + torch_mlir
        MLIR 模型
```

## 安装

```bash
# 从 wheel 文件安装
pip install spiking_mlir-0.1.0-py3-none-any.whl

# 或从源码构建
cd spiking_mlir
pip wheel . --wheel-dir dist --no-deps
pip install dist/spiking_mlir-0.1.0-py3-none-any.whl
```

## 项目结构

```
spiking_mlir/
├── src/spiking_mlir/
│   ├── __init__.py      # 包入口
│   ├── trans.py         # 层替换和导出逻辑
│   └── register.py      # 自定义 ONNX 算子注册
├── pyproject.toml       # 包配置
└── Readme               # 说明文档
```

## 使用方法

### 简单模型

```python
import torch
from spikingjelly.activation_based import layer, neuron, surrogate
from spiking_mlir import import_onnx_model

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        return self.layer(x)

model = SimpleModel()
import_onnx_model(model, input_shape=(1, 1, 28, 28), output_path="model.onnx")
```

### 复杂模型（含时间维度操作）

SpikingJelly 的典型 SNN 模型在 forward 中包含时间维度操作：

```python
class CSNN(torch.nn.Module):
    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.conv_fc = torch.nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            # ... 更多层
        )

    def forward(self, x):
        # 时间维度操作
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N,C,H,W] -> [T,N,C,H,W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)  # [T,N,...] -> [N,...]
        return fr
```

**问题：** `unsqueeze`、`repeat`、`mean` 这些操作会被记录到 ONNX 中，但我们只想导出 `conv_fc` 的计算图。

**解决方案：** 使用 `core_layer` 参数指定要导出的核心网络层：

```python
model = CSNN(T=100, channels=32)

# 指定 core_layer 为 conv_fc，跳过 forward 中的时间维度操作
import_onnx_model(
    model, 
    input_shape=(1, 1, 28, 28), 
    output_path="model.onnx",
    mlir_output_path="model.mlir",
    core_layer=model.conv_fc  # 关键：指定核心层
)
```

### core_layer 参数说明

| 参数类型 | 说明 | 示例 |
|---------|------|------|
| `nn.Module` | 单个模块 | `core_layer=model.conv_fc` |
| `list[nn.Module]` | 多个模块按顺序执行 | `core_layer=[model.conv_fc, model.linear]` |
| `None` | 使用整个模型的 forward | 不传此参数 |

**注意：** 当输入维度 > 4 时（如 `[T, N, C, H, W]`），会自动将前几个维度合并为 batch 维度。

## API 参考

### import_onnx_model

```python
def import_onnx_model(
    model: nn.Module,           # SpikingJelly 模型
    input_shape: tuple,         # 输入形状，如 (1, 1, 28, 28)
    output_path: str,           # ONNX 输出路径
    mlir_output_path: str = None,  # MLIR 输出路径（可选）
    core_layer = None           # 核心网络层（可选）
) -> nn.Module
```

## 层替换规则

### Layer 层

| SpikingJelly 层 | 替换为 |
|----------------|--------|
| layer.Flatten | nn.Flatten |
| layer.Linear | nn.Linear (保留权重) |
| layer.Conv2d | nn.Conv2d (保留权重) |
| layer.Conv1d | nn.Conv1d (保留权重) |
| layer.MaxPool2d | nn.MaxPool2d |
| layer.AvgPool2d | nn.AvgPool2d |
| layer.BatchNorm2d | nn.BatchNorm2d (保留权重，初始化 running stats) |
| layer.Dropout | nn.Dropout |

### Neuron 层

| SpikingJelly 神经元 | 替换为 | 关键参数 |
|-------------------|--------|---------|
| neuron.IFNode | IFNodeWrapper (调用 HcaSnn_ifnode_op) | v_threshold, v_reset |
| neuron.LIFNode | LIFNodeWrapper (调用 HcaSnn_lifnode_op) | v_threshold, v_reset, tau, decay_input |
| neuron.ParametricLIFNode | PLIFNodeWrapper (调用 HcaSnn_plifnode_op) | v_threshold, v_reset, w (可学习参数) |

#### 神经元参数说明

| 参数 | 说明 |
|------|------|
| v_threshold | 发放阈值电压 |
| v_reset | 复位电压 |
| tau | LIF 神经元的时间常数（膜电位衰减速率） |
| decay_input | 输入是否参与衰减（True: 先衰减后加输入，False: 先加输入后衰减） |
| w | ParametricLIFNode 的可学习时间常数参数 |

#### Reset 模式

SpikingJelly 支持两种 reset 模式：

| 模式 | v_reset | 行为 |
|------|---------|------|
| Hard Reset | `v_reset = 0.0, 0.5` 等具体值 | 发放脉冲后电压重置为 v_reset |
| Soft Reset | `v_reset = None` | 发放脉冲后电压减去 v_threshold |

## 常见问题与解决方案

### 1. Meta Kernel 未实现错误

**错误信息：**
```
NotImplementedError: HcaSnn_ops::HcaSnn_lifnode_op: attempted to run this operator with Meta tensors
```

**原因：** PyTorch 2.0+ 的 ONNX 导出需要自定义算子提供 Meta kernel 实现，用于类型推断。

**解决方案：** 在 `register.py` 中为每个自定义算子注册 Meta 实现：
```python
@torch.library.impl("HcaSnn_ops", "Meta")
def meta_ifnode_op(x, v_threshold, v_reset):
    return torch.empty_like(x)
```

### 2. BatchNorm2d 导出后产生 !torch.none

**错误信息：**
```
!torch.none  # MLIR 中的类型错误
UserWarning: Found a node without a valid type proto
```

**原因：** 
- `track_running_stats=False` 时，ONNX 会将 BatchNorm 展开为完整的计算图（ReduceMean、Sub、Mul 等）
- 展开后的中间节点缺少类型信息，导致 torch_mlir 无法推断类型

**解决方案：**
1. 保持 `track_running_stats=True`
2. 初始化 `running_mean=0` 和 `running_var=1`
3. 在 MLIR 导出前运行 ONNX shape inference

```python
# trans.py
new_layer = nn.BatchNorm2d(..., track_running_stats=True)
nn.init.zeros_(new_layer.running_mean)
nn.init.ones_(new_layer.running_var)

# MLIR 导出时
from onnx import shape_inference
model_proto = shape_inference.infer_shapes(model_proto)
```

### 3. 权重复制时 AttributeError

**错误信息：**
```
AttributeError: 'NoneType' object has no attribute 'copy_'
```

**原因：** 当 `track_running_stats=False` 时，`running_mean` 和 `running_var` 为 `None`，直接调用 `copy_()` 会报错。

**解决方案：** 在复制权重前检查属性是否存在：
```python
def copy_weights(source, target):
    if (hasattr(source, 'running_mean') and source.running_mean is not None and
        hasattr(target, 'running_mean') and target.running_mean is not None):
        target.running_mean.copy_(source.running_mean)
        target.running_var.copy_(source.running_var)
```

### 4. ParametricLIFNode 参数名称问题

**错误信息：**
```
TypeError: ParametricLIFNode.__init__() got an unexpected keyword argument 'tau'
AttributeError: 'ParametricLIFNode' object has no attribute 'tau'
```

**原因：** SpikingJelly 的 ParametricLIFNode 使用 `init_tau` 作为初始化参数，内部存储为 `w` 参数。

**解决方案：**
```python
# 初始化时使用 init_tau
neuron.ParametricLIFNode(init_tau=2.0)

# 访问参数时使用 w
tau_value = node.w  # 不是 node.tau
```

### 5. ONNX 导出失败

**错误信息：**
```
RuntimeError: ONNX 导出失败，可能原因：
  1. core_layer 未正确指定，导致导出了包含时间维度操作的完整 forward
  2. 模型包含不支持导出的操作
```

**原因：** 模型的 forward 方法包含 `unsqueeze`、`repeat`、`mean` 等时间维度操作，这些操作不适合直接导出。

**解决方案：** 使用 `core_layer` 参数指定要导出的核心网络层：
```python
import_onnx_model(model, input_shape=(1, 1, 28, 28), output_path="model.onnx", core_layer=model.conv_fc)
```

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- spikingjelly
- onnx
- onnxscript
- torch_mlir (可选，用于 MLIR 导出)
