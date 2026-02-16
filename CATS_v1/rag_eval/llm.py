# rag_eval/llm.py
import json
import logging
import os
from typing import Any, List, Optional

import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from .utils import load_model, load_vllm
import json
from .judge_prompts import nli_prompt
#from .logging_config import logge (removed)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0

        if args.azure_openai_api:
            logger.info("Loading OpenAI model...")
            self._initialize_azure_openai_api(args)
        elif args.vllm:
            logger.info("Loading VLLM model...")
            self._initialize_vllm(args)
        else:
            logger.info("Loading custom model...")
            self._initialize_custom_model(args)

    def _initialize_azure_openai_api(self, args: Any) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        load_dotenv()

        api_key=os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            logger.error("OPENAI_API_KEY not set in environment or .env")
        else:
            logger.info("Using standard OpenAI API client")

        # Standard OpenAI client (NOT Azure)
        self.client = OpenAI(api_key=api_key)

        self.async_client = AsyncOpenAI(api_key=api_key)


    def _initialize_vllm(self, args: Any) -> None:
        self.chat_llm, self.tokenizer, self.sampling_params = load_vllm(
            args.model, args
        )

    def _initialize_custom_model(self, args: Any) -> None:
        self.model, self.tokenizer = load_model(args.model, lora_path=args.lora_path)

    def _call_azure_openai_api(
        self,
        use_azure_chat_completion: bool,
        formatted_prompt: Any,
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> Optional[str]:
        """
        Handles retries and API requests for both `ChatCompletion` and `Completion`.
        """
        retry_count = 0

        while retry_count < 3:
            try:
                if use_azure_chat_completion:
                    response = self.client.chat.completions.create(  # type: ignore[assignment]
                        model=self.args.model,  # model = "deployment_name".
                        messages=formatted_prompt,
                        temperature=self.args.temperature,
                        max_tokens=max_tokens,
                        top_p=self.args.top_p,
                        stop=stop,
                    )
                    answer = response.choices[0].message.content
                else:
                    response = self.client.completions.create(  # type: ignore[assignment]
                        model=self.args.model,
                        prompt=formatted_prompt,
                        temperature=self.args.temperature,
                        max_tokens=max_tokens,
                        top_p=self.args.top_p,
                        stop=["\n", "\n\n"] + (stop or []),
                    )
                    answer = response.choices[0].text  # type: ignore[attr-defined]

                # Update token usage
                self.prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
                self.completion_tokens += getattr(
                    response.usage, "completion_tokens", 0
                )

                return answer

            except Exception as error:
                last_error = error
                retry_count += 1
                logger.warning(f"OpenAI API retry {retry_count} times ({error})")

                if "triggering Azure OpenAI's content management policy" in str(error):
                    self.azure_filter_fail += 1
                    return ""

        print(f"\n Fatal error: {last_error} \n")
        return ""


    def generate(
        self, prompt: str, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Optional[str]:
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning(
                "Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter"
            )
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning(
                "The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter"
            )

        if args.azure_openai_api:
            # Use standard OpenAI chat completions
            try:
                response = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers with grounded, concise text.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.args.temperature,
                    max_tokens=max_tokens,
                    top_p=self.args.top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"generate() OpenAI error: {e}")
                return ""

        else:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=(
                    self.model.config.eos_token_id
                    if isinstance(self.model.config.eos_token_id, list)
                    else [self.model.config.eos_token_id]
                ),
            )
            generation = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
            )
            return generation.strip()

    def batch_generate(self, prompts: List[str]) -> Optional[List[str]]:
        args = self.args

        if args.vllm:
            inputs = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in prompts
            ]
            self.sampling_params.n = (
                1  # Number of output sequences to return for the given prompt
            )
            if isinstance(
                self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id, list
            ):
                self.sampling_params.stop_token_ids = []
                self.sampling_params.stop_token_ids.extend(
                    self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id
                )
            elif isinstance(
                self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id, int
            ):
                self.sampling_params.stop_token_ids = [
                    self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id
                ]

            outputs = self.chat_llm.generate(
                inputs,
                self.sampling_params,
                use_tqdm=True,
            )
            generation = [output.outputs[0].text.strip() for output in outputs]
            return generation

        else:
            raise NotImplementedError("No implemented batch generation method!")
        
    # def judge_behavior(self, prompt: str) -> dict:
    #     """
    #     Call the judge model with a strict JSON prompt.
    #     Returns: {"adherent": bool, "rationale": str}
    #     """
    #     text = self._call_text_api(prompt)  # reuse your existing raw call
    #     try:
    #         start, end = text.find("{"), text.rfind("}")
    #         obj = json.loads(text[start:end+1])
    #         return {
    #             "adherent": bool(obj.get("adherent")),
    #             "rationale": (obj.get("rationale") or "").strip(),
    #         }
    #     except Exception as e:
    #         logger.warning(f"judge_behavior parse failed: {e}")
    #         return {"adherent": False, "rationale": "parse error"}
    
    def judge_behavior(self, prompt: str) -> dict:
        """
        Sync wrapper for backward compatibility.
        """
        try:
            return asyncio.run(self.ajudge_behavior(prompt))
        except RuntimeError:
            # In case we're already in an event loop (e.g. Jupyter),
            # fall back to the old synchronous behavior if you want.
            text = self._call_text_api(prompt)
            try:
                start, end = text.find("{"), text.rfind("}")
                obj = json.loads(text[start:end+1])
                return {
                    "adherent": bool(obj.get("adherent")),
                    "rationale": (obj.get("rationale") or "").strip(),
                }
            except Exception as e:
                logger.warning(f"judge_behavior (sync fallback) parse failed: {e}")
                return {"adherent": False, "rationale": "parse error"}


    def nli_entailment(self, premise: str, hypothesis: str) -> str:
        """
        Decide if premise entails / contradicts / is neutral to hypothesis.
        Returns: "entails" | "contradicts" | "neutral"
        """
        if getattr(self, "use_llm_for_nli", True):
            text = self._call_text_api(nli_prompt(premise, hypothesis))
            try:
                start, end = text.find("{"), text.rfind("}")
                obj = json.loads(text[start:end+1])
                rel = obj.get("relation", "neutral")
                if rel not in ("entails", "contradicts", "neutral"):
                    return "neutral"
                return rel
            except Exception:
                return "neutral"
        else:
            # fallback: local NLI model, e.g. TRUE-NLI
            return "neutral"
            
    def _call_text_api(self, prompt: str) -> str:
        """
        Generic helper used by judge_behavior() and nli_entailment().
        Sends a chat completion request and returns the raw text output.
        """

        # If we are using the OpenAI API branch
        if getattr(self.args, "azure_openai_api", False):
            try:
                response = self.client.chat.completions.create(
                    model=self.args.model,  # e.g. "gpt-4o-mini" or "gpt-4o"
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an evaluation judge. Respond ONLY with JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"_call_text_api OpenAI error: {e}")
                return ""

        # Fallback: if for some reason we are not in the OpenAI branch,
        # just return empty. (You can later extend this to use local HF models.)
        logger.warning("_call_text_api called without OpenAI API configured.")
        return ""
    
    async def _acall_text_api(self, prompt: str) -> str:
        """
        Async helper used by ajudge_behavior() and anli_entailment().
        Sends a chat completion request and returns the raw text output.
        """

        if getattr(self.args, "azure_openai_api", False):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.args.model,  # e.g. "gpt-4o-mini"
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an evaluation judge. Respond ONLY with JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"_acall_text_api OpenAI error: {e}")
                return ""

        logger.warning("_acall_text_api called without OpenAI API configured.")
        return ""
    
    async def ajudge_behavior(self, prompt: str) -> dict:
        """
        Async version of judge_behavior.
        Returns: {"adherent": bool, "rationale": str}
        """
        text = await self._acall_text_api(prompt)
        try:
            start, end = text.find("{"), text.rfind("}")
            obj = json.loads(text[start:end+1])
            return {
                "adherent": bool(obj.get("adherent")),
                "rationale": (obj.get("rationale") or "").strip(),
            }
        except Exception as e:
            logger.warning(f"ajudge_behavior parse failed: {e} | text={text!r}")
            return {"adherent": False, "rationale": "parse error"}
        
    async def anli_entailment(self, premise: str, hypothesis: str) -> str:
        """
        Async NLI: returns 'entails' | 'contradicts' | 'neutral'
        """
        if getattr(self, "use_llm_for_nli", True):
            text = await self._acall_text_api(nli_prompt(premise, hypothesis))
            try:
                start, end = text.find("{"), text.rfind("}")
                obj = json.loads(text[start:end+1])
                rel = obj.get("relation", "neutral")
                if rel not in ("entails", "contradicts", "neutral"):
                    return "neutral"
                return rel
            except Exception:
                return "neutral"
        else:
            return "neutral"

    async def agenerate(
        self, prompt: str, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Optional[str]:
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning(
                "Prompt exceeds max length and return an empty string as answer."
            )
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning(
                "The model can at most generate < 50 tokens. Consider shorter prompts."
            )

        if args.azure_openai_api:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.args.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers with grounded, concise text.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.args.temperature,
                    max_tokens=max_tokens,
                    top_p=self.args.top_p,
                    stop=stop,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"agenerate() OpenAI error: {e}")
                return ""
        else:
            # local HF model: still sync, unless you offload it to a thread/executor
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=(
                    self.model.config.eos_token_id
                    if isinstance(self.model.config.eos_token_id, list)
                    else [self.model.config.eos_token_id]
                ),
            )
            generation = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True
            )
            return generation.strip()