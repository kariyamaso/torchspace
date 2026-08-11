"""Reference demo models (spec §15): one pathological net that makes the
±Z axes earn their keep, and one skip-connection CNN that exercises
branching, nesting and repeated blocks."""
from __future__ import annotations

import torch
from torch import nn


class PathologicalMLP(nn.Module):
    """Deep sigmoid MLP with bad init: vanishing gradients toward the input,
    one hot layer (activation spike) in the middle, and a dying-ReLU stage."""

    def __init__(self, width: int = 64, depth: int = 24):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            lin = nn.Linear(width, width)
            with torch.no_grad():
                lin.weight.mul_(3.0)           # tanh chain -> gradual vanish
                if i == 2:
                    lin.weight.mul_(250.0)     # hot layer -> activation spike
                                               # + saturation wall for grads
            layers += [lin, nn.Sigmoid()]
        self.backbone = nn.Sequential(*layers)
        self.dying = nn.Sequential(nn.LayerNorm(width),
                                   nn.Linear(width, width), nn.ReLU())
        with torch.no_grad():                  # bias < 0 -> mostly-dead ReLU
            self.dying[1].bias.fill_(-0.9)
        self.head = nn.Linear(width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.dying(z)
        return self.head(z)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm2d(cout))
                     if (stride != 1 or cin != cout) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)            # residual add


class TinyResNet(nn.Module):
    """ResNet-18-shaped micro network: stem + 2 stages x 2 blocks + head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU())
        self.layer1 = nn.Sequential(BasicBlock(16, 16), BasicBlock(16, 16))
        self.layer2 = nn.Sequential(BasicBlock(16, 32, 2), BasicBlock(32, 32))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.pool(z).flatten(1)
        return self.fc(z)
