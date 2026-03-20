"""Model client for inference — loads model once, keeps it on GPU."""

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


class ModelClient:
    """Local inference client. Loads the policy once and reuses it across runs."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_id = model_id

        print(f"Loading model {model_id} onto {self.device}...")
        self.model = SmolVLAPolicy.from_pretrained(model_id)
        self.model.to(self.device)

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        print(f"Model loaded and ready on {self.device}.")

    def predict(
        self,
        observation: dict,
        task: str,
        robot_type: str,
        dataset_features: dict,
    ) -> dict:
        """Run one inference step. Returns action dict like {"shoulder_pan.pos": 0.5, ...}."""
        obs_frame = build_inference_frame(
            observation=observation,
            ds_features=dataset_features,
            device=self.device,
            task=task,
            robot_type=robot_type,
        )
        obs = self.preprocess(obs_frame)
        action = self.model.select_action(obs)
        action = self.postprocess(action)
        return make_robot_action(action, dataset_features)

    def reset(self):
        """Flush the internal action queue. Call when switching tasks."""
        self.model.reset()
