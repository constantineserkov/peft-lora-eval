import argparse


parser = argparse.ArgumentParser(description="Training script for a model")

# add arguments
parser.add_argument("--config-path", type=str, default=r"configs\base_config.yaml", help="Path to config YAML file")
parser.add_argument("--method", type=str, default="lora", help="Method name (LoRA/QLoRA/QDoRA)")

# parse arguments
args = parser.parse_args()

# use arguments
print(f"Config path {args.config_path} for {args.method} method")