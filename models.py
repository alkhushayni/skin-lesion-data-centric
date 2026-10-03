"""Architectures: baseline CNN, ResNet50, InceptionV3, MobileNetV2, EfficientNet-B0 and ViT-B/16.

Every pretrained network gets a dropout layer before its 7-class output layer (needed for BALD
Monte-Carlo dropout); `head(model)` returns that final linear layer (used for core-set features).
"""
import torch.nn as nn
from torchvision import models as tvm


class SimpleCNN(nn.Module):
    """Baseline CNN: three convolutional blocks followed by fully connected layers."""

    def __init__(self, num_classes):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(7))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 7 * 7, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes, pretrained=True):
    W = lambda w: w if pretrained else None
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "resnet50":
        m = tvm.resnet50(weights=W(tvm.ResNet50_Weights.IMAGENET1K_V1))
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, num_classes))
        return m
    if name == "inceptionv3":
        m = tvm.inception_v3(weights=W(tvm.Inception_V3_Weights.IMAGENET1K_V1), aux_logits=True,
                             init_weights=not pretrained)
        m.aux_logits, m.AuxLogits = False, None          # single head; has its own dropout before fc
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "mobilenetv2":
        m = tvm.mobilenet_v2(weights=W(tvm.MobileNet_V2_Weights.IMAGENET1K_V1))
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=W(tvm.EfficientNet_B0_Weights.IMAGENET1K_V1))
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "vit_b16":
        m = tvm.vit_b_16(weights=W(tvm.ViT_B_16_Weights.IMAGENET1K_V1))
        m.heads.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.heads.head.in_features, num_classes))
        return m
    raise ValueError(f"unknown model {name}")


def head(model):
    """The final linear layer (its input is the penultimate feature vector)."""
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    return last


DISPLAY_NAMES = {"cnn": "Baseline CNN", "resnet50": "ResNet50", "inceptionv3": "InceptionV3",
                 "mobilenetv2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "vit_b16": "ViT-B/16"}
