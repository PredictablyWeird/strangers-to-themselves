# Behavior-prediction evaluation

Each method is scored at its **tuned best setting** (frozen in `tuning.json`), on the **dev** split. Correlation primary; constant predictions = 0; macro over model×eval cells. Evals: tau2_policy; models: 6; methods: 8. Cells dropped (no data): 0.

## Tuned settings used

| Method | tau2_policy |
|---|---|
| behavioral_sampling | behavioral_sampling |
| cot_flip | cot_flip |
| cross_model_mean | cross_model_mean |
| list_experiment | list_experiment-n6 |
| llm_prediction | llm_prediction |
| pairwise | pairwise-indiff |
| self_report | self_report |
| value | value-context |

## Per method — global (dev)

| Method | mean dev r | 95% CI | cells |
|---|---|---|---|
| behavioral_sampling | -0.076 | [-0.18, +0.02] | 6 |
| cot_flip | -0.081 | [-0.29, +0.12] | 6 |
| cross_model_mean | +0.313 | [+0.16, +0.48] | 6 |
| list_experiment | +0.033 | [-0.19, +0.34] | 6 |
| llm_prediction | +0.014 | [-0.10, +0.13] | 6 |
| pairwise | -0.069 | [-0.24, +0.09] | 6 |
| self_report | +0.233 | [+0.02, +0.51] | 6 |
| value | +0.097 | [+0.01, +0.21] | 6 |

## Per method × eval (mean over models)

| Method | tau2_policy | mean |
|---|---|---|
| behavioral_sampling | -0.08 | -0.08 |
| cot_flip | -0.08 | -0.08 |
| cross_model_mean | +0.31 | +0.31 |
| list_experiment | +0.03 | +0.03 |
| llm_prediction | +0.01 | +0.01 |
| pairwise | -0.07 | -0.07 |
| self_report | +0.23 | +0.23 |
| value | +0.10 | +0.10 |

## Per method × model (mean over evals)

| Method | deepseek-v4-flash-low | gemini-3.1-flash-lite-low | gpt-5.4-nano-low | llama-3.3-70b | llama-4-maverick | qwen3.7-plus-low | mean |
|---|---|---|---|---|---|---|---|
| behavioral_sampling | +0.00 | -0.11 | +0.00 | +0.09 | -0.14 | -0.29 | -0.08 |
| cot_flip | +0.00 | -0.19 | -0.21 | +0.30 | -0.49 | +0.11 | -0.08 |
| cross_model_mean | +0.00 | +0.48 | +0.07 | +0.38 | +0.38 | +0.57 | +0.31 |
| list_experiment | +0.00 | -0.16 | +0.81 | -0.30 | +0.06 | -0.21 | +0.03 |
| llm_prediction | +0.00 | -0.12 | -0.21 | +0.16 | +0.26 | -0.01 | +0.01 |
| pairwise | +0.00 | +0.24 | -0.42 | +0.03 | -0.03 | -0.24 | -0.07 |
| self_report | +0.00 | +0.14 | +0.00 | +0.35 | +0.91 | +0.00 | +0.23 |
| value | +0.00 | -0.02 | -0.04 | +0.20 | +0.12 | +0.31 | +0.10 |

## Per model — best method per eval (dev)

| Model | tau2_policy | mean |
|---|---|---|
| deepseek-v4-flash-low | +0.00 (behavioral_sampling) | +0.00 |
| gemini-3.1-flash-lite-low | +0.48 (cross_model_mean) | +0.48 |
| gpt-5.4-nano-low | +0.81 (list_experiment) | +0.81 |
| llama-3.3-70b | +0.38 (cross_model_mean) | +0.38 |
| llama-4-maverick | +0.91 (self_report) | +0.91 |
| qwen3.7-plus-low | +0.57 (cross_model_mean) | +0.57 |

