import open_clip
import torch.nn as nn


class ConvNextCLIPImageEncoder(nn.Module):
    def __init__(
        self,
        clip_model_name,
        clip_pretrained,
    ):
        super().__init__()

        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained

        self.init_weights()

    def init_weights(self):
        self.clip_pretrained_model = open_clip.create_model(
            model_name=self.clip_model_name, pretrained=self.clip_pretrained
        )

    def forward(self, inputs):
        features, last_features = self.clip_pretrained_model.encode_image(inputs)
        return features, last_features
