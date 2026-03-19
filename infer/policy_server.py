"""Start a PolicyServer for async inference using a YAML config file."""

import argparse

import yaml

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Start async inference policy server")
    parser.add_argument(
        "--config",
        type=str,
        default="config/infer_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    server_cfg = cfg["server"]

    config = PolicyServerConfig(
        host=server_cfg["host"],
        port=server_cfg["port"],
    )

    print(f"Starting policy server on {server_cfg['host']}:{server_cfg['port']}")
    serve(config)


if __name__ == "__main__":
    main()
