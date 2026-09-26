import os
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

def create_openai_model_client():
    """创建并配置 OpenAI 模型客户端"""
    model = os.getenv("LLM_MODEL_ID", "gpt-4o")

    client_kwargs = {
        "model": model,
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    }

    model_info = None

    model_info = {
        "function_calling": True,
        "max_tokens": 4096,
        "context_length": 32768,
        "vision": False,
        "json_output": True,
        "family": "ollama",
        "structured_output": True,
    }

    if not model.startswith(("gpt-", "o1", "o3", "o4")) and not model_info:
        model_info: ModelInfo = {
            "vision": False,            # 是否支持图像输入
            "function_calling": False,  # 是否支持工具/函数调用（本项目未用工具）
            "json_output": False,       # 是否保证可输出 JSON
            "family": "unknown",        # 模型族标识，非官方模型填 unknown
        }

    client_kwargs["model_info"] = model_info

    return OpenAIChatCompletionClient(**client_kwargs)
