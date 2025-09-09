import os
from huggingface_hub import snapshot_download
import os
from pathlib import Path
import yaml
from fastapi import HTTPException
import sentry_sdk
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent


def get_configuration():
    with open(f"{BASE_DIR}/conf/config.yaml", "r") as FH:
        configs = yaml.safe_load(FH)

    return configs


def download_model(hf_model_id: str, model_dir: str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(hf_model_id, local_dir=model_dir, revision="main")
        print(f"Model downloaded successfully! : {model_dir}")
    else:
        print(f"Model already exists! : {model_dir}")


# find path of a file in a given directory
def find_file_path(dir_path, file_name):
    for root, _, files in os.walk(dir_path):
        if file_name in files:
            return root

    return None


def get_model_config(model_alias: str, model_path: str):
    # IMPO: To load huggingface models from custom/local directory,
    # path of config.json is required.
    config_path = find_file_path(model_path, "config.json")
    if not config_path:
        sentry_sdk.capture_message("config.json not found for model: {self.model_path}")
        # Mask the model name before sending it to the client.
        raise HTTPException(
            status_code=500, detail=f"config.json not found for model: {model_alias}"
        )

    return config_path


def get_current_utc_time_and_day():
    utc_now = datetime.now(timezone.utc)

    # ISO 8601 format
    iso_date = utc_now.date().isoformat()

    day_name = utc_now.strftime("%A")

    return iso_date, day_name
