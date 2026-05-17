import PIL
from lmdeploy import api
from lmdeploy import TurbomindEngineConfig
from app_settings import get_env_or_config, get_int_env_or_config


backend_config = TurbomindEngineConfig(
    tp=get_int_env_or_config("LMDEPLOY_TP", ("runtime", "lmdeploy_tp"), 1),
    max_batch_size=4,
    # This parameter is also important for managing concurrent users.
    # It controls the percentage of GPU memory allocated for the K/V cache.
    # A higher value allows more concurrent conversations.

)

client = api.serve(
    model_name=get_env_or_config(
        "INTERNVL_MODEL_NAME",
        ("models", "internvl_model_id"),
        "InternVL2-8B",
    ).split("/")[-1],
    model_path=get_env_or_config(
        "INTERNVL_MODEL_ID",
        ("models", "internvl_model_id"),
        "OpenGVLab/InternVL2-8B",
    ),
    server_name="0.0.0.0",
    server_port=get_int_env_or_config("LMDEPLOY_PORT", ("runtime", "lmdeploy_port"), 23333),
    backend_config=backend_config
)


while True:
    client
 
