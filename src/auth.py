import os
import wandb
from dotenv import load_dotenv
from huggingface_hub import login as hf_login, HfFolder


def check_wandb_api_key(logger):
    # load wandb api key
    load_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")

    # 1. Check env var
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=api_key)
        return True

    # 2. Check default WandB settings file
    settings_path = os.path.expanduser("~/.config/wandb/settings")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            for line in f:
                if line.startswith("api_key:") and line.split(":", 1)[1].strip():
                    return True

    # 3. If no key found suggest to log in
    if not "WANDB_API_KEY" in os.environ:
        logger.info("No WandB API key found. Please log in to WandB.")
        try:
            wandb.login()
            logger.info("WandB login successful")
            return True
        except wandb.errors.UsageError:
            logger.error("WandB API key not set; skipping logging")
            return False
    return None


def init_wandb(config, logger):
    # W&B initialization if enabled
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        if check_wandb_api_key(logger):
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config,
                dir="./results",
            )
            logger.info("W&B initialized.")
        else:
            logger.warning("WandB API key not set; skipping logging")
    else:
        logger.warning("W&B logging disabled.")



def check_hf_token(logger):
    # 1. Check environment variable
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        hf_login(hf_token)
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
    if not hf_token:
        logger.info("No Hugging Face token found. Please log in.")
        try:
            hf_login()  # temporary
            logger.info("Hugging Face login successful")
            return True
        except Exception as e:
            logger.error(f"Hugging Face token not set; skipping model download. Exception:\n\n{e}")
            return False
    return None

def init_hf_auth(logger):
    if check_hf_token(logger):
        logger.info("Hugging Face authentication ready.")
    else:
        logger.warning("Proceeding without Hugging Face token. Gated models will be inaccessible.")
