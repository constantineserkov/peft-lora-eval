import os
import wandb
from huggingface_hub import login as hf_login, HfFolder

from src.logger import get_logger

logger = get_logger()


def check_wandb_api_key():
    # 1. Check env var
    if "WANDB_API_KEY" in os.environ:
        return True

    # 2. Check default WandB settings file
    settings_path = os.path.expanduser("~/.config/wandb/settings")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            for line in f:
                if line.startswith("api_key:") and line.split(":", 1)[1].strip():
                    return True

    # 3. If no key found suggest to log in
    logger.info("No WandB API key found. Please log in to WandB.")
    try:
        wandb.login()
        logger.info("WandB login successful")
    except wandb.errors.UsageError:
        logger.error("WandB API key not set; skipping logging")
        return False


def init_wandb(config):
    # W&B initialization if enabled
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        if check_wandb_api_key():
            wandb.init(
                project=config["project_name"],
                config=config
            )
            logger.info("W&B initialized.")
        else:
            logger.warning("WandB API key not set; skipping logging")
    else:
        logger.warning("W&B logging disabled.")



def check_hf_token():
    # 1. Check environment variable
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return True

    # 2. Check Hugging Face token in default folder
    try:
        token = HfFolder.get_token()
        if token:
            logger.info("Hugging Face token is set")
            return True
        else:
            logger.info("No Hugging Face token found")
    except Exception as e:
        logger.error(f"Error checking Hugging Face token: {e}")

    # 3. No token found, prompt user to log in
    logger.info("No Hugging Face token found. Please log in.")
    try:
        hf_login("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")  # temporary
        logger.info("Hugging Face login successful")
        return True
    except Exception:
        logger.error("Hugging Face token not set; skipping model download")
        return False


def init_hf_auth():
    if check_hf_token():
        logger.info("Hugging Face authentication ready.")
    else:
        logger.warning("Proceeding without Hugging Face token. Gated models will be inaccessible.")
