# Fine-tunable models on Together AI (measured 2026-07-10)

> **Snapshot (2026-07-10).** Provider capabilities were measured live on that date and have
> since changed (e.g. Qwen2.5-72B was delisted from finetuning); re-check before relying on it.

Everything here was tested live against the API, not read from docs. Together exposes **no endpoint
that lists fine-tunable models** (`/v1/fine-tunes/models`, `/api/finetune/models` etc. all 404), and
the model metadata carries no `supportsLora`-style flag. The only reliable test is to create a
fine-tuning job and cancel it immediately — a cancelled job trains no tokens, so it costs nothing.

```python
job = client.fine_tuning.create(training_file=tiny_file, model=base, lora=True, n_epochs=1,
                                lora_r=8, learning_rate=1e-4, suffix="probe", train_on_inputs=False)
client.fine_tuning.cancel(job.id)          # unsupported models raise 404 before this line
```

## The binding constraint is inference, not tunability

15 of 20 probed models are fine-tunable. But most bases are **dedicated-only** for inference
(2×H100 = $0.18/min, 4×H100 = $0.36/min), and that is where the cost lives: the Llama-3.3-70B
self-prediction run spent ~$40 on endpoint minutes and ~nothing on training.

### There is NO serverless LoRA serving. A finetuned model always needs a dedicated endpoint.

The catalogue lists `-Lora` siblings (`google/gemma-4-31B-it-lora`,
`Qwen/Qwen3-30B-A3B-Instruct-2507-Lora`, `meta-llama/Llama-3.3-70B-Instruct-FP8-Lora`, …) which look
like a serverless-adapter path. **They are not.** Tested end to end with a real 12-example finetune of
`google/gemma-4-31B-it`:

| attempt | result |
|---|---|
| call the finetuned output id directly | `non-serverless model` |
| call `google/gemma-4-31B-it-lora` | `non-serverless model` |
| call `google/gemma-4-31B-it` with `extra_body={"adapters":[{"name":<ft>,"weight":1.0}]}` | **200 OK — but the adapter is silently ignored** |

The last row is the trap. It returns a normal completion, so it looks like it worked. It does not: a
**bogus** adapter name (`nobody/does-not-exist-xyz`) is accepted just as happily, and base-only vs
base+adapter give byte-identical output on a prompt the adapter was explicitly trained on
(`q3` → `'3'`; both return the base's generic reply). Never trust the `adapters` field.

So the cost model for any finetune is: serverless base for behavior collection, **dedicated endpoint
for the tuned model**. Endpoint hardware is what varies:

| model | tuned-model endpoint | vs Llama |
|---|---|---|
| google/gemma-4-31B-it | 2×H100 — **$0.18/min** | half |
| meta-llama/Llama-3.3-70B (selfpred v1/v2) | 4×H100 — $0.36/min | baseline |

## Results

Tunable? = job created (then cancelled). Base serverless? = a plain chat call succeeds rather than
raising `non-serverless model`. Tools = returns `tool_calls` (**required for PropensityBench**).
Non-reasoning = answers directly, or does so once thinking is disabled.

| model | tunable | base serverless | tools | non-reasoning | `-Lora` sibling |
|---|---|---|---|---|---|
| **google/gemma-4-31B-it** | ✅ | ✅ $0.39/$0.97 | ✅ | flag | ✅ (meaningless) |
| moonshotai/Kimi-K2.6 | ✅ | ✅ $1.20/$4.50 | ✅ | flag | — |
| openai/gpt-oss-20b | ✅ | ✅ | ✅ | ❌ can't disable | — |
| meta-llama/Llama-3.3-70B-Instruct-Reference | ✅ | serves as `-Turbo` | ✅ | native | ✅ (FP8) |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | ✅ | ❌ | — | — | ✅ (FP8) |
| google/gemma-3-27b-it | ✅ | ❌ | — | — | ✅ |
| Qwen/Qwen3.6-35B-A3B | ✅ | ❌ | — | — | ✅ |
| Qwen/Qwen3.5-35B-A3B | ✅ | ❌ | — | — | ✅ |
| Qwen/Qwen3-8B | ✅ | ❌ | — | — | ✅ |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | ✅ | ❌ | — | — | ✅ |
| Qwen/Qwen3-Next-80B-A3B-Instruct | ✅ | ❌ | — | — | — |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | ✅ | ❌ | — | — | ✅ |
| Qwen/Qwen2.5-72B-Instruct | ✅ | ❌ (`-Turbo` also 400s) | — | — | — |
| Qwen/Qwen2.5-7B-Instruct | ✅ | ❌ | — | — | — |
| zai-org/GLM-4.7 | ✅ | ❌ | — | — | — |
| deepseek-ai/DeepSeek-V3.1 | ✅ | ❌ | — | — | — |
| meta-llama/Meta-Llama-3.1-8B-Instruct-Reference | ✅ | ❌ | — | — | — |

**Not fine-tunable** (404 `not a fine-tunable model`): `Qwen/Qwen3-235B-A22B-Instruct-2507`
(both `-tput` and `-FP8`), `Qwen/Qwen3-4B-Instruct-2507`, `google/gemma-2-27b-it`,
`mistralai/Mistral-Small-24B-Instruct-2501`, `mistralai/Ministral-3-14B-Instruct-2512`,
`MiniMaxAI/MiniMax-M3`.

Qwen3-235B is the frustrating case: serverless, tool-calling, non-reasoning natively, $0.20/$0.60 —
and untunable. It is still a good **pool addition** as a base model.

## Reasoning is on by default, and `reasoning_effort` does not work

Most modern models on Together emit reasoning unless told otherwise. Together **ignores**
`reasoning_effort` (the key our `models.yaml` uses for OpenRouter models). What works is:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

Verified to flip `gemma-4-31B-it`, `Kimi-K2.6`, `GLM-5.2` and `MiniMax-M3` from empty
`finish_reason='length'` replies to a direct answer. `openai/gpt-oss-*` ignores both and cannot be
run as a non-reasoning model. `inspect_ai` forwards `GenerateConfig.extra_body` to Together, so this
is usable from `models.yaml` once `common.resolve_model` passes the key through.

Two gotchas that produced wrong answers during this survey:

- Testing tool-calling **without** disabling thinking makes a model look like it lacks tool support:
  it spends the token budget reasoning and returns no `tool_calls`. Always disable thinking first.
- `pricing.input > 0` does **not** imply serverless. `Llama-4-Scout`, `GLM-4.7`, `DeepSeek-V3.1`,
  `Mistral-Small-24B` all carry prices and all raise `non-serverless model`. Only a real call tells
  you.

## Serverless availability is narrow

Of 166 chat models, exactly **21** answered a plain serverless call. `running=True` in the catalogue
is not the flag (it is `False` for everything), and neither is a nonzero price.

## Recommendation

`google/gemma-4-31B-it` — tunable, serverless base (so behavior collection is endpoint-free and
costs a couple of dollars), tool-calling, runnable non-reasoning, and its tuned endpoint is 2×H100 at
half Llama's rate. No serverless LoRA exists for anything, so that is as cheap as a finetune gets
here.

Second choice `moonshotai/Kimi-K2.6` for lab diversity, at a larger endpoint.

## Training/inference template mismatch (open)

Together renders the chat template at training time with thinking **on** by default, while we infer
with `enable_thinking=false`. Check whether that degrades a tuned gemma before trusting its numbers;
if it does, bake the non-thinking rendering into the training data.
