"""统一 LLM 客户端 — 替代 coze_coding_dev_sdk.LLMClient / ImageGenerationClient

使用 OpenAI SDK，兼容所有 OpenAI 格式的 API (OpenAI / DeepSeek / 通义千问 / 智谱 等)。
通过环境变量配置:
    OPENAI_API_KEY   — API 密钥
    OPENAI_BASE_URL  — API 地址 (默认 https://api.openai.com/v1)
    OPENAI_MODEL     — 模型名 (默认 gpt-4o)

生图单独配置（不设置则复用 LLM 的 OPENAI_* 配置）:
    IMAGE_GEN_BASE_URL  — 生图 API 地址
    IMAGE_GEN_API_KEY   — 生图 API 密钥
    IMAGE_GEN_MODEL     — 生图模型名
"""

import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── 已知不支持生图的服务商（用于友好报错） ──
_NO_IMAGE_PROVIDERS = ["deepseek", "zhipu", "qwen", "tongyi", "moonshot", "minimax", "doubao", "volces"]


def get_llm_client() -> OpenAI:
    """获取 LLM 客户端实例"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def get_image_client() -> OpenAI:
    """获取生图专用客户端 — 优先用 IMAGE_GEN_* 环境变量，未设置则复用 LLM 配置"""
    api_key = os.getenv("IMAGE_GEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("IMAGE_GEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_image_model() -> str:
    """获取生图模型名"""
    return os.getenv("IMAGE_GEN_MODEL", "dall-e-3")


def _is_image_supported(client: OpenAI) -> str:
    """检查当前 provider 是否可能支持生图。返回空字符串表示支持，否则返回原因。"""
    base_url = str(client.base_url).lower()
    for kw in _NO_IMAGE_PROVIDERS:
        if kw in base_url:
            return f"当前 LLM 服务商不支持图片生成。请在环境变量中设置 IMAGE_GEN_BASE_URL / IMAGE_GEN_API_KEY / IMAGE_GEN_MODEL 指向支持生图的服务"
    return ""


def call_llm(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    model: str = None,
    max_tokens: int = 8192,
) -> str:
    """统一 LLM 调用，返回文本字符串。

    替代 LLMClient.invoke() — 接口兼容旧代码。
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = get_llm_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""


def call_llm_langchain(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    model: str = None,
    max_tokens: int = 8192,
) -> str:
    """LLM 调用（LangChain 消息格式）— 兼容旧代码中使用 SystemMessage/HumanMessage 的场景。

    直接接收文本 prompt，内部转成 OpenAI 格式。
    """
    return call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    )


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    model: str = None,
    style: str = "vivid",
) -> list[str]:
    """AI 生图。

    优先使用 IMAGE_GEN_* 环境变量指定的服务商；未设置则尝试复用 LLM 配置。
    返回图片 URL 列表。服务商不支持时返回空列表（不抛异常）。
    """
    client = get_image_client()

    # 检查 provider 是否支持生图
    not_supported = _is_image_supported(client)
    if not_supported:
        logger.warning(not_supported)
        return []

    # 映射尺寸
    size_map = {"2K": "1024x1024", "4K": "1792x1024", "1K": "512x512"}
    mapped_size = size_map.get(size, "1024x1024")
    model_name = model or get_image_model()

    try:
        response = client.images.generate(
            model=model_name,
            prompt=f"女装牛仔裤, {prompt}, 高质量商品展示图, 商业摄影级别",
            size=mapped_size,
            style=style,
            n=1,
        )
        return [img.url for img in response.data if img.url]
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "not found" in err_msg.lower() or "not_found" in err_msg.lower():
            logger.warning(f"Image generation not supported by this API provider: {e}")
            return []
        logger.error(f"Image generation failed: {e}")
        raise
