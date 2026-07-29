# qwen_llama_e2e_sft_baseline_balanced_4reviewers

- Seed: `20260715`
- Selected unique sample-variants: `350`
- Total review assignments: `700`
- Extra 30-sample cells: `qwen7b|minimal|baseline, llama8b|runtime|sft`

## Selected Pool

- Conflict counts: `{'1': 70, '2': 70, '3': 70, '4': 70, '5': 70}`
- Model counts: `{'llama8b': 175, 'qwen7b': 175}`
- Train counts: `{'baseline': 175, 'sft': 175}`
- Prompt counts: `{'minimal': 117, 'runtime': 117, 'strict': 116}`
- Cell counts: `{'llama8b|minimal|baseline': 29, 'llama8b|minimal|sft': 29, 'llama8b|runtime|baseline': 29, 'llama8b|runtime|sft': 30, 'llama8b|strict|baseline': 29, 'llama8b|strict|sft': 29, 'qwen7b|minimal|baseline': 30, 'qwen7b|minimal|sft': 29, 'qwen7b|runtime|baseline': 29, 'qwen7b|runtime|sft': 29, 'qwen7b|strict|baseline': 29, 'qwen7b|strict|sft': 29}`
- Base-id duplication: `{'max_occurrence': 2, 'duplicated_base_ids': 33}`

## Pair Quotas

| Pair | Target | Actual |
| --- | ---: | ---: |
| atharv / manan | 83 | 83 |
| atharv / parth | 84 | 84 |
| atharv / samyek | 33 | 33 |
| manan / parth | 83 | 83 |
| manan / samyek | 34 | 34 |
| parth / samyek | 33 | 33 |

## Reviewer Summary

### manan

- Total: `200`
- Conflict: `{'1': 40, '2': 40, '3': 40, '4': 40, '5': 40}`
- Model: `{'llama8b': 100, 'qwen7b': 100}`
- Train: `{'baseline': 100, 'sft': 100}`
- Prompt: `{'minimal': 67, 'runtime': 66, 'strict': 67}`
- Repeated base ids: `{'count': 4, 'max_repeat': 2}`
- Cell mix: `{'llama8b|minimal|baseline': 16, 'llama8b|minimal|sft': 15, 'llama8b|runtime|baseline': 17, 'llama8b|runtime|sft': 20, 'llama8b|strict|baseline': 17, 'llama8b|strict|sft': 15, 'qwen7b|minimal|baseline': 19, 'qwen7b|minimal|sft': 17, 'qwen7b|runtime|baseline': 14, 'qwen7b|runtime|sft': 15, 'qwen7b|strict|baseline': 17, 'qwen7b|strict|sft': 18}`

### atharv

- Total: `200`
- Conflict: `{'1': 40, '2': 40, '3': 40, '4': 40, '5': 40}`
- Model: `{'llama8b': 100, 'qwen7b': 100}`
- Train: `{'baseline': 100, 'sft': 100}`
- Prompt: `{'minimal': 67, 'runtime': 67, 'strict': 66}`
- Repeated base ids: `{'count': 5, 'max_repeat': 2}`
- Cell mix: `{'llama8b|minimal|baseline': 18, 'llama8b|minimal|sft': 14, 'llama8b|runtime|baseline': 18, 'llama8b|runtime|sft': 16, 'llama8b|strict|baseline': 15, 'llama8b|strict|sft': 19, 'qwen7b|minimal|baseline': 17, 'qwen7b|minimal|sft': 18, 'qwen7b|runtime|baseline': 17, 'qwen7b|runtime|sft': 16, 'qwen7b|strict|baseline': 15, 'qwen7b|strict|sft': 17}`

### parth

- Total: `200`
- Conflict: `{'1': 40, '2': 40, '3': 40, '4': 40, '5': 40}`
- Model: `{'llama8b': 100, 'qwen7b': 100}`
- Train: `{'baseline': 100, 'sft': 100}`
- Prompt: `{'minimal': 67, 'runtime': 67, 'strict': 66}`
- Repeated base ids: `{'count': 5, 'max_repeat': 2}`
- Cell mix: `{'llama8b|minimal|baseline': 15, 'llama8b|minimal|sft': 19, 'llama8b|runtime|baseline': 15, 'llama8b|runtime|sft': 17, 'llama8b|strict|baseline': 19, 'llama8b|strict|sft': 15, 'qwen7b|minimal|baseline': 15, 'qwen7b|minimal|sft': 18, 'qwen7b|runtime|baseline': 19, 'qwen7b|runtime|sft': 16, 'qwen7b|strict|baseline': 17, 'qwen7b|strict|sft': 15}`

### samyek

- Total: `100`
- Conflict: `{'1': 20, '2': 20, '3': 20, '4': 20, '5': 20}`
- Model: `{'llama8b': 50, 'qwen7b': 50}`
- Train: `{'baseline': 50, 'sft': 50}`
- Prompt: `{'minimal': 33, 'runtime': 34, 'strict': 33}`
- Repeated base ids: `{'count': 0, 'max_repeat': 1}`
- Cell mix: `{'llama8b|minimal|baseline': 9, 'llama8b|minimal|sft': 10, 'llama8b|runtime|baseline': 8, 'llama8b|runtime|sft': 7, 'llama8b|strict|baseline': 7, 'llama8b|strict|sft': 9, 'qwen7b|minimal|baseline': 9, 'qwen7b|minimal|sft': 5, 'qwen7b|runtime|baseline': 8, 'qwen7b|runtime|sft': 11, 'qwen7b|strict|baseline': 9, 'qwen7b|strict|sft': 8}`
