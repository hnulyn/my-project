"""
Global path configuration module
Automatically detects runtime environment based on current working directory and configures data paths
"""

from pathlib import Path
import os


class GlobalConfig:
    """Global configuration class that automatically configures paths based on runtime environment"""
    
    def __init__(self):
        self.working_dir = Path.cwd().__str__()
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup paths based on working directory"""
 
        self.data_base_dir = os.environ.get("DATA_BASE_DIR", "./data")
        self.project_base_dir = os.environ.get("PROJECT_BASE_DIR", ".")
    
        self.averitec_data_dir = os.path.join(self.data_base_dir, "AVeriTeC")
        self.hero_data_dir = os.path.join(self.data_base_dir, "HerO", "data_store", "baseline")
        
        self.debatecv_dir = os.path.join(self.project_base_dir, "debatecv")
        self.result_base_dir = os.path.join(self.working_dir, "output")
        self.temp_dir = os.path.join(self.result_base_dir, "temp")
        
        os.makedirs(self.result_base_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def get_averitec_data_path(self, split: str = "train") -> str:
        """Get AVeriTeC data file path"""
        if split not in ["train", "dev"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'")
        return os.path.join(self.averitec_data_dir, f"{split}.json")
    
    def get_hero_data_path(self, split: str = "train") -> str:
        """Get HerO data file path"""
        if split not in ["train", "dev"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'")
        return os.path.join(self.hero_data_dir, f"{split}_veracity_prediction_8b.json")
    
    def get_output_dir(self, experiment_name: str = None) -> str:
        """Get output directory path"""
        if experiment_name:
            return os.path.join(self.result_base_dir, experiment_name)
        return self.result_base_dir
    
    def print_config(self):
        """Print current configuration"""
        pass


# Create global config instance
global_config = GlobalConfig()


def get_data_base_dir() -> str:
    """Get data base directory"""
    return global_config.data_base_dir


def get_averitec_data_dir() -> str:
    """Get AVeriTeC data directory"""
    return global_config.averitec_data_dir


def get_hero_data_dir() -> str:
    """Get HerO data directory"""
    return global_config.hero_data_dir


def get_output_base_dir() -> str:
    """Get output base directory"""
    return global_config.result_base_dir


def get_temp_dir() -> str:
    """Get temp directory"""
    return global_config.temp_dir
