"""Shared immutable constants for provider and endpoint configuration."""

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE_OPENAI = "azure_openai"

ALLOWED_CHAT_PROVIDERS = (
    PROVIDER_OPENROUTER,
    PROVIDER_OPENAI,
    PROVIDER_AZURE_OPENAI,
)

OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
