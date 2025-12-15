from typing import Any

import torch
import torch.nn.functional as F
import timm
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VisionBackboneExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        channels, height, width = observation_space.shape
        # Temporary value, will be updated after backbone is created
        super().__init__(observation_space, features_dim=1)

        self.backbone = timm.create_model(
            "hf_hub:timm/mobilenetv3_small_100.lamb_in1k",
            pretrained=True,
            in_chans=channels,
            features_only=True,
            out_indices=[-1],
        )
        self.output_dim = self.backbone.feature_info[-1]["num_chs"]

        self._features_dim = self.output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        features = self.backbone(x)[0]
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return pooled


class VisionBackbonePolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["features_extractor_class"] = VisionBackboneExtractor
        super().__init__(*args, **kwargs)
