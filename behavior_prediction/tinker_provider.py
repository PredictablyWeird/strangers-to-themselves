"""inspect_ai model provider for Tinker-hosted models (base models or LoRA sampler weights).

Registers the ``tinker`` provider so ``models.yaml`` entries can point at either a Tinker
base model or a finetuned checkpoint (see ``scripts/finetune_introspection.py``):

    qwen3-30b-a3b:
      model: tinker/Qwen/Qwen3-30B-A3B-Instruct-2507
    qwen3-30b-a3b-intro:
      model: tinker/tinker://<run-id>/sampler_weights/<name>

Model names starting with ``tinker://`` are opened as sampler-weights paths, anything else
as a base model. Prompts are rendered with the base tokenizer's chat template
(``enable_thinking=False`` — harmless for non-thinking models, disables the <think> block
for hybrid Qwen models). Registration happens on import (guarded in the package
``__init__``); the ``tinker`` SDK is a .venv-only extra, not a package dependency, and
``TINKER_API_KEY`` comes from the environment/.env like the other provider keys.

Limitations: text-only chat (no tools, images, or logprobs) — enough for
``common.elicit_rates``, which is the only way the benchmark talks to models.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelOutput, modelapi
from inspect_ai.tool import ToolChoice, ToolInfo


@modelapi(name="tinker")
class TinkerAPI(ModelAPI):
    def __init__(self, model_name: str, base_url: str | None = None,
                 api_key: str | None = None, api_key_vars: list[str] = [],
                 config: GenerateConfig = GenerateConfig()) -> None:
        super().__init__(model_name, base_url, api_key,
                         api_key_vars=["TINKER_API_KEY"], config=config)
        self._client: Any = None
        self._tokenizer: Any = None

    def _ensure_client(self) -> None:
        if self._client is None:
            import tinker

            sc = tinker.ServiceClient()
            if self.model_name.startswith("tinker://"):
                self._client = sc.create_sampling_client(model_path=self.model_name)
            else:
                self._client = sc.create_sampling_client(base_model=self.model_name)
            self._tokenizer = self._client.get_tokenizer()

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo],
                       tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        from tinker import types

        if tools:
            raise NotImplementedError("tinker provider does not support tool calling")
        self._ensure_client()

        messages = [{"role": m.role, "content": m.text} for m in input]
        prompt_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=False, enable_thinking=False)

        params: dict[str, Any] = {"max_tokens": config.max_tokens or 1024}
        for key in ("temperature", "top_p", "top_k", "seed"):
            value = getattr(config, key)
            if value is not None:
                params[key] = value
        if config.stop_seqs:
            params["stop"] = config.stop_seqs

        response = await self._client.sample_async(
            prompt=types.ModelInput.from_ints(prompt_ids), num_samples=1,
            sampling_params=types.SamplingParams(**params))
        seq = response.sequences[0]
        text = self._tokenizer.decode(seq.tokens, skip_special_tokens=True)
        stop_reason = "stop" if str(seq.stop_reason) == "stop" else "max_tokens"
        return ModelOutput.from_content(model=self.model_name, content=text,
                                        stop_reason=stop_reason)
