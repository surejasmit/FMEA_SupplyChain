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
import signal
from functools import wraps

logger = logging.getLogger(__name__)

# Resource limits
MAX_BATCH_SIZE = 1000
MAX_TEXT_LENGTH = 10000
LLM_TIMEOUT_SECONDS = 30

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(seconds):
    """Decorator to add timeout to function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set alarm signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable alarm
            return result
        return wrapper
    return decorator


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
        """
        Extract FMEA components from text using LLM with retry logic
        
        Args:
            text: Input text (review, report, complaint)

        Returns:
            Dictionary with extracted information:
            {
                'failure_mode': str,
                'effect': str,
                'cause': str,
                'component': str,
                'existing_controls': str
            }
        """
        # Enforce text length limit
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated from {len(text)} to {MAX_TEXT_LENGTH} characters")
            text = text[:MAX_TEXT_LENGTH]
        
        if self.pipeline is None:
            return self._rule_based_extraction(text)

        try:
<<<<<<< fix/Uncontrolled_Resource_Consumption_DoS_Vulnerability
            prompt = self._build_extraction_prompt(text)
            response = self._generate_llm_response_with_timeout(prompt)
            extracted_info = self._parse_llm_response(response)
            
            # Validate extraction
            if self._is_valid_extraction(extracted_info):
                return self._validate_extraction(extracted_info)
            else:
                # Log and retry with stricter prompt
                self._log_extraction_failure(text, response, "Invalid extraction - retrying")
                raise ValueError("Invalid extraction format")
                
        except (TimeoutError, Exception) as e:
            logger.warning(f"First extraction attempt failed: {e}")
            
        # Retry attempt with stricter prompt  
        try:
            strict_prompt = self._build_strict_retry_prompt(text)
            response = self._generate_llm_response_with_timeout(strict_prompt)
            extracted_info = self._parse_llm_response(response)
            
            if self._is_valid_extraction(extracted_info):
                return self._validate_extraction(extracted_info)
            else:
                self._log_extraction_failure(text, response, "Retry also failed")
                raise ValueError("Retry extraction also invalid")
                
        except (TimeoutError, Exception) as e:
            logger.error(f"Both extraction attempts failed: {e}")
            self._log_extraction_failure(text, "", f"Complete failure: {e}")
            return self._rule_based_extraction(text)
    
    def _build_extraction_prompt(self, text: str) -> str:
        """
        Build structured extraction prompt with clear definitions and examples
        
        Args:
            text: Input text to analyze
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in Failure Mode and Effects Analysis (FMEA). 
Analyze the following text and extract failure-related information.

IMPORTANT DEFINITIONS:
- FAILURE MODE: What physically fails or deviates from normal operation (e.g., "Gearbox oil leakage", "Motor overheating")
- EFFECT: The consequence experienced by the end user or downstream system (e.g., "Reduced transmission life", "System shutdown")
- CAUSE: The root originating mechanism that triggers the failure (e.g., "Seal ring damage", "Inadequate ventilation")
- COMPONENT: The specific physical system or subsystem affected (e.g., "Gearbox", "Power Supply")

RULES:
1. If the cause is genuinely unknown, output "Unknown" rather than guessing
2. Distinguish failure mode (what breaks) from effect (what happens after)
3. Output ONLY valid JSON with no explanation or preamble

EXAMPLES:

Input: "The hydraulic cylinder is leaking fluid due to a damaged seal ring. This is causing the hydraulic system to fail."
Output: {{"failure_mode": "Hydraulic cylinder leakage", "effect": "Hydraulic system failure", "cause": "Seal ring damage", "component": "Hydraulic cylinder"}}

Input: "Motor temperature is extremely high, probably due to bearing damage or fan failure. The motor will be damaged if this continues."
Output: {{"failure_mode": "Excessively high motor temperature", "effect": "Motor damage", "cause": "Bearing damage or motor fan damage", "component": "Motor"}}

Input: "The gas turbine starter has low efficiency and fails to turn the engine when starting."
Output: {{"failure_mode": "Function failure when not turning the engine on starting", "effect": "Function failure", "cause": "Starter low efficiency", "component": "Gas turbine starter"}}

Now analyze this text:
=======
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

>>>>>>> main
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
<<<<<<< fix/Uncontrolled_Resource_Consumption_DoS_Vulnerability
            do_sample=False,  # Use deterministic generation for consistency
            temperature=0.1   # Low temperature for factual extraction
        )[0]['generated_text']
        return response.strip()
    
    def _generate_llm_response_with_timeout(self, prompt: str) -> str:
        """
        Generate LLM response with timeout protection
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated response text
            
        Raises:
            TimeoutError: If generation exceeds timeout
        """
        try:
            # Use signal-based timeout (Unix-like systems)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(LLM_TIMEOUT_SECONDS)
            try:
                response = self._generate_llm_response(prompt)
            finally:
                signal.alarm(0)
            return response
        except AttributeError:
            # Windows doesn't support SIGALRM, use threading timeout
            import threading
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = self._generate_llm_response(prompt)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=LLM_TIMEOUT_SECONDS)
            
            if thread.is_alive():
                logger.error("LLM inference timeout")
                raise TimeoutError("LLM inference exceeded timeout")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
    
    def _is_valid_extraction(self, extracted: Dict[str, str]) -> bool:
        """
        Validate if extraction contains required fields
        
        Args:
            extracted: Extracted information dictionary
            
        Returns:
            True if valid extraction
        """
        required_fields = ['failure_mode', 'effect', 'cause', 'component']
        
        # Check all required fields exist
        for field in required_fields:
            if field not in extracted:
                return False
            
            value = extracted[field]
            if not value or value.lower().strip() in ['', 'unknown', 'none', 'n/a', 'not specified']:
                # Allow "Unknown" for cause field only (as per rules)
                if field == 'cause' and value.lower().strip() == 'unknown':
                    continue
                return False
        
        return True
    
    def _log_extraction_failure(self, input_text: str, model_response: str, reason: str):
        """
        Log extraction failures for debugging
        
        Args:
            input_text: Original input text
            model_response: Model's response
            reason: Failure reason
        """
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'input': input_text[:500],  # Truncate long inputs
            'model_response': model_response[:500],
            'reason': reason
        }
        
        log_file = log_dir / "extraction_failures.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM response to extract structured information with improved parsing
        
        Args:
            response: Raw LLM output

        Returns:
            Parsed dictionary
        """
        # Clean the response
        response = response.strip()
        
        try:
            # Try direct JSON parsing first
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Try to find JSON block in response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
                    
            # Try to extract JSON from code blocks
            code_block_pattern = r'```(?:json)?\s*({.*?})\s*```'
            code_matches = re.findall(code_block_pattern, response, re.DOTALL)
            
            for match in code_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        # Fallback: try to extract key-value pairs
        result = {
            'failure_mode': 'Unknown',
            'effect': 'Unknown',
            'cause': 'Unknown',
            'component': 'Unknown'
        }
        
        # Try various patterns to extract fields
        patterns = {
            'failure_mode': [r'["\']?failure_mode["\']?\s*[:=]\s*["\']([^"\'\\n}]+)["\']?', 
                           r'failure mode:\s*([^\\n}]+)', 
                           r'what fails\?\s*:\s*([^\\n}]+)'],
            'effect': [r'["\']?effect["\']?\s*[:=]\s*["\']([^"\'\\n}]+)["\']?',
                      r'effect:\s*([^\\n}]+)',
                      r'consequence:\s*([^\\n}]+)'],
            'cause': [r'["\']?cause["\']?\s*[:=]\s*["\']([^"\'\\n}]+)["\']?',
                     r'cause:\s*([^\\n}]+)',
                     r'reason:\s*([^\\n}]+)'],
            'component': [r'["\']?component["\']?\s*[:=]\s*["\']([^"\'\\n}]+)["\']?',
                         r'component:\s*([^\\n}]+)',
                         r'system:\s*([^\\n}]+)']
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().rstrip(',"\'')
                    if value and value.lower() not in ['unknown', 'none', 'n/a']:
                        result[field] = value
                    break
        
        return result

    def _validate_extraction(self, extracted: Dict[str, str]) -> Dict[str, str]:
        """
        Validate and clean extracted information

        Args:
            extracted: Raw extracted dictionary

        Returns:
            Validated dictionary with cleaned values
        """
        required_keys = ['failure_mode', 'effect', 'cause', 'component']
        
        # Ensure all required keys exist
        for key in required_keys:
            if key not in extracted:
                extracted[key] = 'Not specified'
        
        # Clean and validate each field
        for key, value in extracted.items():
            if not value or str(value).strip() == '':
                extracted[key] = 'Not specified'
            elif str(value).lower().strip() in ['none', 'n/a', 'not applicable']:
                extracted[key] = 'Not specified'
            else:
                # Clean the value
                cleaned = str(value).strip()
                # Remove quotes if present
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                if cleaned.startswith("'") and cleaned.endswith("'"):
                    cleaned = cleaned[1:-1]
                extracted[key] = cleaned
        
        # Add existing_controls if not present for backward compatibility
        if 'existing_controls' not in extracted:
            extracted['existing_controls'] = 'Not specified'
        
        return extracted

    def _rule_based_extraction(self, text: str) -> Dict[str, str]:
        """
        Fallback rule-based extraction when LLM is unavailable

        Args:
            text: Input text

        Returns:
            Extracted information dictionary
        """
        logger.info("Using rule-based extraction")

        text_lower = text.lower()

        # Identify failure mode keywords
        failure_keywords = [
            "fail",
            "broke",
            "malfunction",
            "problem",
            "issue",
            "defect",
            "not work",
            "stopped",
            "error",
            "fault",
        ]

        effect_keywords = [
            "result",
            "consequence",
            "impact",
            "caused",
            "led to",
            "unable",
        ]

        cause_keywords = ["because", "due to", "caused by", "reason", "from"]

        component_keywords = [
            "engine",
            "brake",
            "transmission",
            "steering",
            "suspension",
            "electrical",
            "battery",
            "tire",
            "wheel",
            "door",
            "window",
        ]

        # Extract failure mode
        failure_mode = self._extract_with_keywords(text, failure_keywords)

        # Extract effect
        effect = self._extract_with_keywords(text, effect_keywords)

        # Extract cause
        cause = self._extract_with_keywords(text, cause_keywords)

        # Extract component
        component = self._extract_with_keywords(text, component_keywords)
=======
            do_sample=False,
            temperature=0.1,
        )[0]["generated_text"]
>>>>>>> main

        return response.strip()

    # ---------------- VALIDATION ---------------- #

    def _is_valid(self, data: Dict) -> bool:
        keys = ["failure_mode", "effect", "cause", "component"]
        return all(k in data and data[k] for k in keys)

    def _clean_output(self, data: Dict) -> Dict[str, str]:
        for k, v in data.items():
            if not v:
                data[k] = "Not specified"

<<<<<<< fix/Uncontrolled_Resource_Consumption_DoS_Vulnerability
    def batch_extract(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Extract failure information from multiple texts with size limit
=======
        if "existing_controls" not in data:
            data["existing_controls"] = "Not specified"
>>>>>>> main

        return data

<<<<<<< fix/Uncontrolled_Resource_Consumption_DoS_Vulnerability
        Returns:
            List of extracted information dictionaries
        """
        # Enforce batch size limit
        if len(texts) > MAX_BATCH_SIZE:
            logger.warning(f"Batch size {len(texts)} exceeds limit {MAX_BATCH_SIZE}. Truncating.")
            texts = texts[:MAX_BATCH_SIZE]
        
        logger.info(f"Batch extracting from {len(texts)} texts")

        results = []

        # Use tqdm for progress bar
        for text in tqdm(texts, desc="Extracting FMEA information", unit="text"):
            try:
                extracted = self.extract_failure_info(text)
                results.append(extracted)
            except Exception as e:
                logger.error(f"Failed to extract from text: {e}")
                # Add fallback result to maintain list consistency
                results.append(self._rule_based_extraction(text[:1000]))

        return results
=======
    # ---------------- PARSER ---------------- #

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
>>>>>>> main

        return {
            "failure_mode": "Unknown",
            "effect": "Unknown",
            "cause": "Unknown",
            "component": "Unknown",
        }

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