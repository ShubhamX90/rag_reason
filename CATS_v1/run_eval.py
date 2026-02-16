from rag_eval.evaluator import Evaluator
from rag_eval.config import eval_cfg
from rag_eval.data import load_dataset
from rag_eval.llm import LLM
from types import SimpleNamespace


# Load toy dataset
dataset = load_dataset("finalparse/final_eval_qwen_e2e_sft.jsonl")

# Dummy LLM client (just returns "adherent": true)
class DummyClient:
    def __call__(self, prompt: str) -> str:
        return '{"adherent": true, "rationale": "ok"}'

# Minimal dummy LLM
class DummyLLM:
    def judge_behavior(self, prompt: str):
        return {"adherent": True, "rationale": "always adherent"}

    def nli_entailment(self, premise: str, hypothesis: str):
        return "entails"

#llm = LLM(client=DummyClient())

args = SimpleNamespace(
    # Use the Azure/OpenAI branch in LLM
    azure_openai_api=True,   # set to False if you want to use vLLM or a local HF model
    vllm=False,

    # Model name:
    # - For Azure OpenAI: this must be your *deployment name* (e.g., "gpt-4o-mini")
    # - For normal OpenAI client (if you rewired llm.py): use "gpt-4o" / "gpt-4o-mini" etc.
    model="gpt-4o-mini",

    # Generation params
    temperature=0.0,
    top_p=1.0,

    # LoRA / adapter path if using custom HF model (not needed for OpenAI)
    lora_path=None,
)

llm = LLM(args)

# Run evaluation

evaluator = Evaluator(eval_cfg, llm)
results = evaluator.evaluate(dataset)
print(results)