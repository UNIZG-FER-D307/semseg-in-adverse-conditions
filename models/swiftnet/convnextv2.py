import timm
import torch.nn as nn


def convnextv2_tiny_timm(model_name, pretrained=False, checkpoint_path=None):
    assert (
        not pretrained or not checkpoint_path
    ), f"Use local weights or pretrained, not both"

    return timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path="" if not checkpoint_path else checkpoint_path,
        features_only=True,
    )


def convnextv2_large_timm(model_name, pretrained=False, checkpoint_path=None):
    assert (
        not pretrained or not checkpoint_path
    ), f"Use local weights or pretrained, not both"

    return timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path="" if not checkpoint_path else checkpoint_path,
        features_only=True,
    )
