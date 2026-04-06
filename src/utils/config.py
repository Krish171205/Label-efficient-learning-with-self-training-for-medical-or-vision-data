"""
Configuration loader.

Loads YAML config files and provides easy access to hyperparameters.
"""

import os
import yaml
from typing import Any


class Config:
    """
    Loads a YAML config and provides dot-notation + dict access.
    
    Usage:
        cfg = Config("configs/default.yaml")
        print(cfg.training.backbone)       # "resnet18"
        print(cfg["training"]["epochs"])   # 50
    """
    
    def __init__(self, path: str = None, data: dict = None):
        if path is not None:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        self._data = data
        
        # Convert nested dicts to Config objects for dot access
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(data=value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> dict:
        return self._data
    
    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(config_path: str = None) -> Config:
    """
    Load config from YAML file. Falls back to default.yaml if no path given.
    
    Args:
        config_path: Path to YAML config file.
    
    Returns:
        Config object with all hyperparameters.
    """
    if config_path is None:
        # Find default.yaml relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(project_root, "configs", "default.yaml")
    
    cfg = Config(config_path)
    print(f"✓ Config loaded from {config_path}")
    return cfg
