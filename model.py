import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


LETTERS = list("اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
VOCAB = (
    ["<blank>"] + LETTERS + [" ", "‌"] + [chr(0x6F0 + _i) for _i in range(10)] + ["<unk>"]
)
VOCAB_SIZE = len(VOCAB)
TIME_STRIDE = 8

class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        x = torch.relu(self.bn2(self.pw(x)))
        return x


class DSCNNv6(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 128, 3, stride=(2, 2), padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.blocks = nn.Sequential(
            DSBlock(128, 192),
            DSBlock(192, 192, stride=(2, 2)),
            DSBlock(192, 256),
            DSBlock(256, 256, stride=(1, 2)),
            DSBlock(256, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
        )
        self.head = nn.Linear(336, vocab)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn0(self.conv0(x)))
        x = self.blocks(x)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

