"""One-time export of the pretrained UFLD (ResNet-18, CULane) model to ONNX for cv2.dnn.

Needs PyTorch only for this step; run it with a throwaway environment:
    uv run --no-project --with torch --with torchvision --with onnx --with numpy \
        --index https://download.pytorch.org/whl/cpu python scripts/export_ufld_onnx.py
Inputs:  models/ufld_repo (git clone of cfzd/Ultra-Fast-Lane-Detection), models/culane_18.pth
Output:  models/ufld_culane_18.onnx
"""

import sys
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "ufld_repo"))

# The 2020 repo calls torchvision.models.resnet18(pretrained=...); newer torchvision dropped that argument.
_resnet18 = torchvision.models.resnet18
torchvision.models.resnet18 = lambda pretrained=False, **kw: _resnet18(weights=None, **kw)

from model.model import parsingNet  # noqa: E402

net = parsingNet(pretrained=False, backbone="18", cls_dim=(201, 18, 4), use_aux=False)
state = torch.load(ROOT / "models" / "culane_18.pth", map_location="cpu")["model"]
net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=False)
net.eval()

out = ROOT / "models" / "ufld_culane_18.onnx"
torch.onnx.export(net, torch.zeros(1, 3, 288, 800), out, input_names=["image"],
                  output_names=["lanes"], opset_version=11, dynamo=False)
print(f"wrote {out}")
