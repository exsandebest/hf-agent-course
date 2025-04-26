import os
from typing import Any

from smolagents import HfApiModel, InferenceClientModel, LiteLLMModel, OpenAIServerModel


def get_huggingface_api_model(model_id: str, **kwargs) -> Any:
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

    return HfApiModel(model_id=model_id, token=api_key, **kwargs)


def get_inference_client_model(model_id: str, **kwargs) -> Any:
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

    return InferenceClientModel(model_id=model_id, token=api_key, **kwargs)


def get_openai_server_model(model_id: str, **kwargs) -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    api_base = os.getenv("OPENAI_API_BASE")
    if not api_base:
        raise ValueError("OPENAI_API_BASE is not set")

    return OpenAIServerModel(
        model_id=model_id, api_key=api_key, api_base=api_base, **kwargs
    )


def get_lite_llm_model(model_id: str, **kwargs) -> Any:
    return LiteLLMModel(model_id=model_id, **kwargs)


def get_model(model_type: str, model_id: str, **kwargs) -> Any:

    models = {
        "HfApiModel": get_huggingface_api_model,
        "InferenceClientModel": get_inference_client_model,
        "OpenAIServerModel": get_openai_server_model,
        "LiteLLMModel": get_lite_llm_model,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](model_id, **kwargs)
