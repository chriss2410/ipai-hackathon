"""Model client for inference — loads model once, keeps it on GPU."""

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


class ModelClient:
    """Local inference client. Loads the policy once and reuses it across runs.

    For LoRA models, pass lora_adapter_id in addition to model_id:
      - model_id: the base model (e.g. "chris241094/smolVLA-New-v07")
      - lora_adapter_id: the adapter checkpoint (e.g. "chris241094/smolVLA-New-lora-v09")
    The adapter config already embeds the base model path, so if only lora_adapter_id
    is provided the base model is discovered automatically.
    """

    def __init__(self, model_id: str, device: str = "cpu", lora_adapter_id: str | None = None):
        self.device = torch.device(device)
        self.model_id = model_id
        self.lora_adapter_id = lora_adapter_id

        if lora_adapter_id:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(lora_adapter_id)
            # Prefer the explicitly supplied base model_id; fall back to the one
            # embedded in the adapter config (peft_config.base_model_name_or_path).
            base_id = model_id or peft_config.base_model_name_or_path
            print(f"Loading base model {base_id} onto {self.device}...")
            base_model = SmolVLAPolicy.from_pretrained(base_id)
            print(f"Applying LoRA adapter {lora_adapter_id}...")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_id, config=peft_config)
            pretrained_path = lora_adapter_id
        else:
            print(f"Loading model {model_id} onto {self.device}...")
            self.model = SmolVLAPolicy.from_pretrained(model_id)
            pretrained_path = model_id

        self.model.to(self.device)

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            pretrained_path,
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
