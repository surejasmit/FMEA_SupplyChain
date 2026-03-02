"""
LLM-Based Information Extraction Module
Uses transformer models to extract FMEA-relevant information from text
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from typing import Dict, List
import json
import re
import logging
import datetime
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LLMExtractor:
    """
    Uses LLM to extract failure mode, effect, cause, and related information from text
    """

    # SECURITY: Whitelist of trusted models
    TRUSTED_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "google/flan-t5-base",
        "google/flan-t5-large",
    ]

    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.prompts = config.get("prompts", {})

        self.model = None
        self.tokenizer = None
        self.pipeline = None

        self._load_model()

    # ---------------- SECURITY ---------------- #

    def _validate_model_name(self, model_name: str) -> bool:
        """Validate model name against whitelist"""
        return model_name in self.TRUSTED_MODELS

    # ---------------- MODEL LOADING ---------------- #

    def _load_model(self):
        model_name = self.model_config.get(
            "name", "mistralai/Mistral-7B-Instruct-v0.2"
        )

        if not self._validate_model_name(model_name):
            logger.error(f"Model '{model_name}' not trusted. Using rule-based extraction.")
            self.pipeline = None
            return

        logger.info(f"Loading model: {model_name}")

        try:
            # Quantization
            quant_config = None
            if self.model_config.get("quantization", True):
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            # Tokenizer (SECURE)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=False
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Device handling
            device = self.model_config.get("device", "auto")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Model (SECURE)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                trust_remote_code=False,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )

            self.model = self.model.to(device)

            # Pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.model_config.get("max_length", 512),
                temperature=self.model_config.get("temperature", 0.3),
                top_p=self.model_config.get("top_p", 0.9),
                do_sample=True,
            )

            logger.info(f"Model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.pipeline = None

    # ---------------- EXTRACTION ---------------- #

    def extract_failure_info(self, text: str) -> Dict[str, str]:
        if self.pipeline is None:
            return self._rule_based_extraction(text)

        try:
            prompt = self._build_prompt(text)
            response = self._generate_response(prompt)
            extracted = self._parse_llm_response(response)

            if self._is_valid(extracted):
                return self._clean_output(extracted)

            raise ValueError("Invalid extraction")

        except Exception as e:
            logger.warning(f"Retry extraction due to error: {e}")

        # Retry strict
        try:
            prompt = self._strict_prompt(text)
            response = self._generate_response(prompt)
            extracted = self._parse_llm_response(response)

            if self._is_valid(extracted):
                return self._clean_output(extracted)

        except Exception as e:
            logger.error(f"Retry failed: {e}")

        return self._rule_based_extraction(text)

    # ---------------- PROMPTS ---------------- #

    def _build_prompt(self, text: str) -> str:
        return f"""
Extract failure information in JSON.

Text: {text}

Output:
"""

    def _strict_prompt(self, text: str) -> str:
        return f"""
Return ONLY JSON with keys:
failure_mode, effect, cause, component.

Text: {text}
"""

    # ---------------- LLM ---------------- #

    def _generate_response(self, prompt: str) -> str:
        response = self.pipeline(
            prompt,
            return_full_text=False,
            do_sample=False,
            temperature=0.1,
        )[0]["generated_text"]

        return response.strip()

    # ---------------- VALIDATION ---------------- #

    def _is_valid(self, data: Dict) -> bool:
        keys = ["failure_mode", "effect", "cause", "component"]
        return all(k in data and data[k] for k in keys)

    def _clean_output(self, data: Dict) -> Dict[str, str]:
        for k, v in data.items():
            if not v:
                data[k] = "Not specified"

        if "existing_controls" not in data:
            data["existing_controls"] = "Not specified"

        return data

    # ---------------- PARSER ---------------- #

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return {
            "failure_mode": "Unknown",
            "effect": "Unknown",
            "cause": "Unknown",
            "component": "Unknown",
        }

    # ================================================================
    # Benchmarking / Ensemble Methods
    # ================================================================

    def run_benchmark_extraction(self, text: str) -> Dict[str, Dict[str, str]]:
        """
        Run extraction using every model registered in config['benchmarking'].
        Each result is tagged with the model id.

        Args:
            text: Input text to analyze

        Returns:
            Dict mapping model_id -> extracted_info dict
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        benchmarking_cfg = self.config.get('benchmarking', {})
        active_models = benchmarking_cfg.get('active_models', [])

        if not active_models:
            logger.warning("No active_models in benchmarking config; using current model only.")
            return {"default": self.extract_failure_info(text)}

        results: Dict[str, Dict[str, str]] = {}

        def _run_single_model(model_cfg: Dict) -> tuple:
            """Run extraction for a single model configuration."""
            model_id = model_cfg.get('id', 'unknown')
            model_path = model_cfg.get('model_path', model_cfg.get('name', ''))
            model_type = model_cfg.get('type', 'local')

            try:
                if model_type == 'rule' or model_path == 'Rule-based (No LLM)':
                    extracted = self._rule_based_extraction(text)
                else:
                    import copy
                    isolated_config = copy.deepcopy(self.config)
                    isolated_config['model']['name'] = model_path
                    temp_extractor = LLMExtractor(isolated_config)
                    extracted = temp_extractor.extract_failure_info(text)
                return (model_id, extracted)
            except Exception as exc:
                logger.error(f"Benchmark extraction failed for {model_id}: {exc}")
                return (model_id, self._rule_based_extraction(text))

        # Parallelise across models using ThreadPoolExecutor
        max_workers = min(len(active_models), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_model, cfg): cfg for cfg in active_models}
            for future in as_completed(futures):
                model_id, extracted = future.result()
                results[model_id] = extracted

        return results

    def run_temperature_sweep_extraction(self, text: str,
                                          temperatures: Optional[List[float]] = None
                                          ) -> Dict[str, Dict[str, str]]:
        """
        Simulate multi-model output using a single model at different temperatures.
        Useful for testing variance analysis before integrating real different models.

        Args:
            text: Input text
            temperatures: List of temperature values (defaults from config)

        Returns:
            Dict mapping 'temp_<value>' -> extracted_info
        """
        if temperatures is None:
            sweep_cfg = self.config.get('benchmarking', {}).get('temperature_sweep', {})
            temperatures = sweep_cfg.get('temperatures', [0.1, 0.5, 0.8])

        results: Dict[str, Dict[str, str]] = {}

        if self.pipeline is None:
            for temp in temperatures:
                results[f"temp_{temp}"] = self._rule_based_extraction(text)
            return results

        for temp in temperatures:
            try:
                prompt = self._build_extraction_prompt(text)
                response = self.pipeline(
                    prompt,
                    max_new_tokens=self.model_config.get('max_length', 512),
                    return_full_text=False,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                )[0]['generated_text'].strip()

                extracted = self._parse_llm_response(response)
                if self._is_valid_extraction(extracted):
                    results[f"temp_{temp}"] = self._validate_extraction(extracted)
                else:
                    results[f"temp_{temp}"] = self._rule_based_extraction(text)
            except Exception as exc:
                logger.warning(f"Temperature sweep ({temp}) failed: {exc}")
                results[f"temp_{temp}"] = self._rule_based_extraction(text)

        return results

    def batch_benchmark_extract(self, texts: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Batch extraction across all benchmark models for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Dict mapping model_id -> list of extracted info dicts
        """
        benchmarking_cfg = self.config.get('benchmarking', {})
        active_models = benchmarking_cfg.get('active_models', [])

        if not active_models:
            return {"default": self.batch_extract(texts)}

        results: Dict[str, List[Dict[str, str]]] = {}

        for model_cfg in active_models:
            model_id = model_cfg.get('id', 'unknown')
            model_path = model_cfg.get('model_path', model_cfg.get('name', ''))
            model_type = model_cfg.get('type', 'local')

            try:
                if model_type == 'rule' or model_path == 'Rule-based (No LLM)':
                    extracted_list = [self._rule_based_extraction(t) for t in texts]
                else:
                    import copy
                    isolated_config = copy.deepcopy(self.config)
                    isolated_config['model']['name'] = model_path
                    temp_extractor = LLMExtractor(isolated_config)
                    extracted_list = temp_extractor.batch_extract(texts)
                results[model_id] = extracted_list
            except Exception as exc:
                logger.error(f"Batch benchmark failed for {model_id}: {exc}")
                results[model_id] = [self._rule_based_extraction(t) for t in texts]

        return results


if __name__ == "__main__":
    # Example usage
    import yaml
    # ---------------- RULE BASED ---------------- #


    def _rule_based_extraction(self, text: str) -> Dict[str, str]:
        logger.info("Rule-based extraction fallback")

        return {
            "failure_mode": text[:80],
            "effect": "Functionality impacted",
            "cause": "Under investigation",
            "component": "General",
            "existing_controls": "Not specified",
        }

    # ---------------- BATCH ---------------- #

    def batch_extract(self, texts: List[str]) -> List[Dict[str, str]]:
        results = []
        for t in tqdm(texts):
            results.append(self.extract_failure_info(t))
        return results