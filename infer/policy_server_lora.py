"""Policy server variant that loads LoRA adapter models correctly.

The default PolicyServer uses `policy_class.from_pretrained()` which loads only
the base model weights. For LoRA checkpoints, the adapter weights (saved separately
as adapter_model.safetensors) must be layered on top of the base model.

This server subclasses PolicyServer and overrides the model-loading step to use
PeftModel, which:
  1. Reads the adapter config to find the base model (e.g. chris241094/smolVLA-New-v07)
  2. Loads the base model
  3. Applies the LoRA adapter weights on top

Usage:
    Terminal 1: python infer/policy_server_lora.py
    Terminal 2: python infer/infer_lora.py
"""

import argparse
import pickle  # nosec
from pathlib import Path

import yaml

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer, serve
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2
from lerobot.async_inference.helpers import RemotePolicyConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class LoRAPolicyServer(PolicyServer):
    """PolicyServer that loads a LoRA adapter on top of its base model."""

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        self.logger.info(
            f"Loading LoRA model | adapter={policy_specs.pretrained_name_or_path} | "
            f"device={policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        # --- LoRA-aware loading ---
        from peft import PeftConfig, PeftModel

        adapter_path = policy_specs.pretrained_name_or_path
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_path = peft_config.base_model_name_or_path

        self.logger.info(f"Base model: {base_model_path}")
        self.logger.info(f"Adapter:    {adapter_path}")

        policy_class = get_policy_class(self.policy_type)
        base_policy = policy_class.from_pretrained(base_model_path)
        self.policy = PeftModel.from_pretrained(base_policy, adapter_path, config=peft_config)
        self.policy.to(self.device)
        # -------------------------

        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=adapter_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        return services_pb2.Empty()


def main():
    parser = argparse.ArgumentParser(description="Start async inference policy server (LoRA)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "infer_config_lora.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    server_cfg = cfg["server"]

    import grpc
    from concurrent import futures
    from lerobot.transport import services_pb2_grpc

    config = PolicyServerConfig(
        host=server_cfg["host"],
        port=server_cfg["port"],
    )

    policy_server = LoRAPolicyServer(config)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{config.host}:{config.port}")

    print(f"Starting LoRA policy server on {config.host}:{config.port}")
    print(f"Adapter: chris241094/smolVLA-New-lora-v09")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
