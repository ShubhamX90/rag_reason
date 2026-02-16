# response_generator.py
# -*- coding: utf-8 -*-
"""
Response Generator
------------------

This module generates model responses for evaluation.

Pipeline:
  1. Load prompts and evaluation data
  2. Construct prompts with demonstrations
  3. Generate responses (single or batch)
  4. Optionally apply post-hoc citation assignment
  5. Save results to JSON

Supports both:
  • vLLM batch inference (fast, GPU-based)
  • Standard sequential generation

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import json
import random
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from .config import ResponseGeneratorConfig
from .llm import LLM
from .logging_config import logger
from .post_hoc_cite import post_hoc_cite
from .utils import make_demo


class ResponseGenerator:
    """
    Generate model responses for evaluation datasets.
    """

    def __init__(self, config: ResponseGeneratorConfig):
        self.config = config
        self.llm = LLM(config)
        self.max_length = self._set_max_length(config.model, config.max_length)
        logger.info(f"Set the model max length to {self.max_length}")

        self.prompt_file = config.prompt_file
        self.data_file = config.data_file
        self.output_path = config.eval_file
        self.load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self, prompt_file: Optional[str] = None, data_file: Optional[str] = None
    ) -> None:
        """
        Load prompt and evaluation data from JSON files.
        """
        prompt_fp = prompt_file or self.prompt_file
        data_fp = data_file or self.data_file
        assert prompt_fp is not None, "Prompt file must be provided."
        assert data_fp is not None, "Data file must be provided."

        self.prompt_data = json.load(open(prompt_fp, "r", encoding="utf-8"))
        all_data = json.load(open(data_fp, "r", encoding="utf-8"))
        # Quick test option for debugging
        self.eval_data = all_data[: self.config.quick_test]

    # ------------------------------------------------------------------
    # Generation pipeline
    # ------------------------------------------------------------------

    def generate_responses(
        self, data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for evaluation data.
        """
        if data is None:
            data = self.eval_data

        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        incomplete_doc_list = 0
        for idx, eval_item in enumerate(tqdm(data, desc="Preparing prompts")):
            eval_item["prompt"] = self.construct_prompt(eval_item)
            doc_list = eval_item.get("docs", [])[: self.config.ndoc]
            eval_item["docs"] = doc_list
            if len(doc_list) < self.config.ndoc:
                incomplete_doc_list += 1

        if incomplete_doc_list > 0:
            logger.warning(
                f"{incomplete_doc_list} questions have incomplete document lists."
            )

        # Batch generation (vLLM) vs single-query
        if self.config.vllm:
            outputs = self._batch_generate(data)
        else:
            outputs = self._sequential_generate(data)

        # Post-hoc citation
        if self.config.posthoc:
            outputs = post_hoc_cite(outputs, self.config)

        self.eval_data = outputs
        return outputs

    def _batch_generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch generation using vLLM.
        """
        prompts = [
            item["prompt"] for item in data for _ in range(self.config.num_samples)
        ]
        prompt_lengths = [
            len(self.llm.tokenizer.tokenize(prompt)) for prompt in prompts
        ]
        max_prompt_len = max(prompt_lengths)
        logger.info(f"Max prompt length: {max_prompt_len}")

        # Run batch inference
        batch_outputs = self.llm.batch_generate(prompts)
        assert batch_outputs is not None

        # Release vLLM memory
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        if hasattr(self.llm.chat_llm.llm_engine.model_executor, "driver_worker"):
            del self.llm.chat_llm.llm_engine.model_executor.driver_worker
        torch.cuda.empty_cache()

        # Post-process per item
        for idx, item in enumerate(tqdm(data, desc="Processing outputs")):
            output_array = []
            for j, output in enumerate(
                batch_outputs[idx : idx + self.config.num_samples]
            ):
                response = output.replace("<|im_end|>", "").rstrip()
                if response.endswith("End."):
                    response = response[: -len("End.")]
                output_array.append(response)
            item["output"] = output_array if len(output_array) > 1 else output_array[0]

        return data

    def _sequential_generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sequential generation query-by-query.
        """
        for idx, item in enumerate(tqdm(data, desc="Sequential generation")):
            prompt = item["prompt"]
            item["output"] = self.generate_response_single_query(prompt)
        return data

    def generate_response_single_query(self, prompt: str) -> Union[str, List[str]]:
        """
        Generate a response for a single query.
        """
        prompt_len = len(self.llm.tokenizer.tokenize(prompt))
        output_array = []

        for _ in range(self.config.num_samples):
            logger.debug(f"Prompt length: {prompt_len}")
            response = self.llm.generate(
                prompt,
                min(self.config.max_new_tokens, self.config.max_length - prompt_len),
            )
            if response is None:
                raise ValueError("Generation failed: LLM returned None.")
            response = response.replace("<|im_end|>", "").rstrip()
            if response.endswith("End."):
                response = response[: -len("End.")]
            output_array.append(response)

        return output_array if len(output_array) > 1 else output_array[0]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def construct_prompt(self, eval_item: Dict[str, Any]) -> str:
        """
        Build a prompt with demonstrations + docs + instructions.
        """
        head_prompt = self._generate_head_prompt()
        return head_prompt + make_demo(
            eval_item,
            prompt=self.prompt_data["demo_prompt"],
            ndoc=self.config.ndoc,
            doc_prompt=self.prompt_data["doc_prompt"],
            instruction=self.prompt_data["instruction"],
            test=True,
        )

    def _generate_head_prompt(self) -> str:
        """
        Build the head prompt from demonstration examples.
        Supports refusal-aware demos or regular demos.
        """
        head_prompt = ""
        if not self.config.no_demo:
            if "refusal" in str(self.config.prompt_file).lower():
                logger.warning("Using refusal head prompts...")
                pos_train_ids = np.random.choice(
                    len(self.prompt_data["positive_demos"]),
                    ceil(self.config.shot / 2),
                    replace=False,
                )
                rej_train_ids = np.random.choice(
                    len(self.prompt_data["refusal_demos"]),
                    self.config.shot // 2,
                    replace=False,
                )
                train_items = [
                    *[self.prompt_data["positive_demos"][i] for i in pos_train_ids],
                    *[self.prompt_data["refusal_demos"][i] for i in rej_train_ids],
                ]
                random.shuffle(train_items)
            else:
                train_ids = np.random.choice(
                    len(self.prompt_data["demos"]), self.config.shot, replace=False
                )
                train_items = [self.prompt_data["demos"][i] for i in train_ids]

            for train_item in train_items:
                head_prompt += self._make_demo_item(train_item)
                head_prompt += self.prompt_data["demo_sep"]

        return head_prompt

    def _make_demo_item(self, train_item: Dict[str, Any]) -> str:
        """
        Format a single demonstration item for the head prompt.
        """
        ndoc = self.config.ndoc
        if self.config.no_doc_in_demo:
            ndoc = 0
        elif self.config.fewer_doc_in_demo:
            assert self.config.ndoc_in_demo is not None
            ndoc = self.config.ndoc_in_demo

        return make_demo(
            train_item,
            prompt=self.prompt_data["demo_prompt"],
            ndoc=ndoc,
            doc_prompt=self.prompt_data["doc_prompt"],
            instruction=self.prompt_data["instruction"],
            use_shorter=None,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_responses(self, output_path: Optional[str] = None) -> None:
        """
        Save generated responses to a JSON file.
        """
        if output_path is not None:
            self.output_path = output_path
        assert self.output_path is not None, "Output path not set."

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"data": self.eval_data, "config": vars(self.config)}, f, indent=4
            )
        logger.info(f"Results saved to {self.output_path}")

    # ------------------------------------------------------------------
    # Max length setup
    # ------------------------------------------------------------------

    def _set_max_length(self, model_name: str, max_len: Optional[int]) -> int:
        """
        Determine max length based on the model type.
        """
        if max_len is not None:
            return max_len
        m = model_name.lower()
        if "16k" in m:
            return 16384
        if "32k" in m:
            return 32768
        if "turbo" in m:
            return 4096
        if "gpt-4" in m or "gpt4" in m:
            return 8192
        if "llama-2" in m or "llama2" in m:
            return 4096
        return 2048  # default fallback
