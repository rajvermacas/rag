"""Router for provider-specific chat backends."""

import logging
from types import MappingProxyType
from typing import AsyncIterator, Mapping

from app.config import ChatBackendProfile
from app.services.chat_provider_models import BackendChatProvider, ChatModelOption, require_non_empty


logger = logging.getLogger(__name__)


class ChatProviderRouter:
    """Routes chat requests by backend id and validated model allowlists."""

    def __init__(
        self,
        backend_profiles: Mapping[str, ChatBackendProfile],
        providers: Mapping[str, BackendChatProvider],
    ) -> None:
        if len(backend_profiles) == 0:
            raise ValueError("backend_profiles must not be empty")
        if len(providers) == 0:
            raise ValueError("providers must not be empty")
        self._backend_profiles = MappingProxyType(dict(backend_profiles))
        self._providers = MappingProxyType(dict(providers))
        self._validate_provider_coverage()

    async def generate_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        profile = self._resolve_profile(backend_id)
        normalized_model = self._validate_model_allowed(profile, model)
        provider = self._resolve_provider(profile.backend_id)
        logger.info(
            "chat_provider_router_generate backend_id=%s provider=%s model=%s",
            profile.backend_id,
            profile.provider,
            normalized_model,
        )
        return await provider.generate_chat_response_with_model(
            model=normalized_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        profile = self._resolve_profile(backend_id)
        normalized_model = self._validate_model_allowed(profile, model)
        provider = self._resolve_provider(profile.backend_id)
        logger.info(
            "chat_provider_router_stream backend_id=%s provider=%s model=%s",
            profile.backend_id,
            profile.provider,
            normalized_model,
        )
        async for chunk in provider.stream_chat_response_with_model(
            model=normalized_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            yield chunk

    def list_model_options(self) -> tuple[ChatModelOption, ...]:
        options: list[ChatModelOption] = []
        for profile in self._backend_profiles.values():
            for model in profile.models:
                options.append(
                    ChatModelOption(
                        backend_id=profile.backend_id,
                        provider=profile.provider,
                        model=model,
                        label=f"{profile.backend_id} ({profile.provider}) \u00b7 {model}",
                    )
                )
        return tuple(options)

    def _validate_provider_coverage(self) -> None:
        missing_providers = [
            backend_id
            for backend_id in self._backend_profiles
            if backend_id not in self._providers
        ]
        if len(missing_providers) > 0:
            raise ValueError(
                "provider instances are missing for backend ids: "
                f"{', '.join(missing_providers)}"
            )

    def _resolve_profile(self, backend_id: str) -> ChatBackendProfile:
        normalized_backend_id = require_non_empty(backend_id, "backend_id")
        if normalized_backend_id not in self._backend_profiles:
            raise ValueError("backend_id is not allowed")
        return self._backend_profiles[normalized_backend_id]

    def _validate_model_allowed(self, profile: ChatBackendProfile, model: str) -> str:
        normalized_model = require_non_empty(model, "model")
        if normalized_model not in profile.models:
            raise ValueError("model is not allowed for backend_id")
        return normalized_model

    def _resolve_provider(self, backend_id: str) -> BackendChatProvider:
        if backend_id not in self._providers:
            raise ValueError(f"provider instance is missing for backend_id: {backend_id}")
        return self._providers[backend_id]
