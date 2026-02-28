"""
Secure YAML Configuration Validator
Prevents YAML injection attacks (CWE-502)
"""

import yaml
import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration against strict schema"""
    
    ALLOWED_MODEL_NAMES = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
    ]
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and values
        
        Args:
            config: Loaded YAML configuration
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Validate model section
        if 'model' in config:
            ConfigValidator._validate_model(config['model'])
        
        # Validate risk_scoring section
        if 'risk_scoring' in config:
            ConfigValidator._validate_risk_scoring(config['risk_scoring'])
        
        # Validate text_processing section
        if 'text_processing' in config:
            ConfigValidator._validate_text_processing(config['text_processing'])
        
        return True
    
    @staticmethod
    def _validate_model(model: Dict[str, Any]):
        """Validate model configuration"""
        if not isinstance(model, dict):
            raise ValueError("model must be a dictionary")
        
        # Validate model name
        if 'name' in model:
            name = model['name']
            if name is not None:  # Allow None for rule-based mode
                if not isinstance(name, str):
                    raise ValueError("model.name must be a string")
                if name not in ConfigValidator.ALLOWED_MODEL_NAMES:
                    raise ValueError(f"model.name '{name}' not in whitelist")
        
        # Validate numeric parameters
        if 'max_length' in model:
            if not isinstance(model['max_length'], int) or not (1 <= model['max_length'] <= 4096):
                raise ValueError("model.max_length must be integer 1-4096")
        
        if 'temperature' in model:
            if not isinstance(model['temperature'], (int, float)) or not (0 <= model['temperature'] <= 2):
                raise ValueError("model.temperature must be number 0-2")
        
        if 'device' in model:
            if model['device'] not in ['auto', 'cpu', 'cuda']:
                raise ValueError("model.device must be 'auto', 'cpu', or 'cuda'")
    
    @staticmethod
    def _validate_risk_scoring(risk_scoring: Dict[str, Any]):
        """Validate risk scoring configuration"""
        if not isinstance(risk_scoring, dict):
            raise ValueError("risk_scoring must be a dictionary")
    
    @staticmethod
    def _validate_text_processing(text_processing: Dict[str, Any]):
        """Validate text processing configuration"""
        if not isinstance(text_processing, dict):
            raise ValueError("text_processing must be a dictionary")
        
        if 'min_review_length' in text_processing:
            if not isinstance(text_processing['min_review_length'], int):
                raise ValueError("text_processing.min_review_length must be integer")


def load_config_safe(config_path: str) -> Dict[str, Any]:
    """
    Safely load and validate YAML configuration
    
    Args:
        config_path: Path to config file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If validation fails
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate against schema
        ConfigValidator.validate_config(config)
        
        logger.info(f"Configuration loaded and validated: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise ValueError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
