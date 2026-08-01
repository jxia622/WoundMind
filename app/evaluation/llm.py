from __future__ import annotations

import asyncio
import json
from typing import TypeVar

from pydantic import BaseModel

from app.agent.config import AgentConfig


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOpenAIClient:
    """Small repository-local adapter for schema-enforced Responses API calls."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.openai_enabled

    async def generate(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        payload: dict,
        model: str | None = None,
    ) -> SchemaT:
        if not self.enabled:
            raise RuntimeError("OpenAI structured evaluation is not configured.")
        return await asyncio.to_thread(
            self._generate_sync,
            schema=schema,
            system_prompt=system_prompt,
            payload=payload,
            model=model or self.config.evaluation_model,
        )

    @staticmethod
    def _generate_sync(
        *,
        schema: type[SchemaT],
        system_prompt: str,
        payload: dict,
        model: str,
    ) -> SchemaT:
        from openai import OpenAI

        response = OpenAI().responses.parse(
            model=model,
            instructions=system_prompt,
            input=json.dumps(payload, indent=2, default=str),
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Structured OpenAI response did not contain parsed output.")
        return parsed
