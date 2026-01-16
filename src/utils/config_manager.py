"""
Configuration management utilities
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration file
        
        Args:
            config_path: Configuration file path
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.config_path = config_path
            return self.config
        except Exception as e:
            raise ValueError(f"Cannot load configuration file {config_path}: {e}")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration file
        
        Args:
            config_path: Configuration file path, uses current path if None
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("Configuration file path not specified")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key, supports dot-separated nested keys
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key, supports dot-separated nested keys
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]) -> None:
        """
        Update configuration
        
        Args:
            other_config: Configuration dictionary to merge
        """
        self._deep_update(self.config, other_config)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return copy of configuration dictionary"""
        return self.config.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """Create configuration manager from dictionary"""
        manager = cls()
        manager.config = config_dict.copy()
        return manager
    
    @classmethod
    def create_default_config(cls) -> 'ConfigManager':
        """Create default configuration"""
        default_config = {
            "models": {
                "debate_model": "gpt-4o-mini",
                "judge_model": "gpt-4o",
                "temperature": 0.9,
                "max_tokens": 512,
                "top_p": 0.7
            },
            "api_settings": {
                "openai_api_key": "",
                "openai_base_url": "",
                "llama_api_key": "",
                "llama_base_url": "",
                "qwen_api_key": "",
                "qwen_base_url": "",
                "sleep_time": 0
            },
            "debate_settings": {
                "max_rounds": 3,
                "num_players": 3,
                "verbose": False
            },
            "prompts": {
                "player_meta_prompt": "",
                "moderator_meta_prompt": "",
                "judge_meta_prompt": "",
                "affirmative_prompt": "",
                "negative_prompt": "",
                "moderator_prompt": "",
                "judge_prompt_last1": "",
                "judge_prompt_last2": "",
                "debate_prompt": ""
            }
        }
        
        return cls.from_dict(default_config)
