import yaml
from pathlib import Path
from smart_pandas.config.data_config import DataConfig


def read_yaml(path: str) -> dict:
    """Read YAML file with proper error handling.
    
    Parameters
    ----------
    path : str
        Path to the YAML file
        
    Returns
    -------
    dict
        Parsed YAML content
        
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    yaml.YAMLError
        If the YAML is malformed
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(file_path) as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")


def read_config(path: str) -> DataConfig:
    """Read and parse a DataConfig from a YAML file.
    
    Parameters
    ----------
    path : str
        Path to the configuration YAML file
        
    Returns
    -------
    DataConfig
        Parsed configuration object
        
    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist
    yaml.YAMLError
        If the YAML is malformed
    ConfigValidationError
        If the configuration is invalid
    """
    try:
        config_data = read_yaml(path)
        
        return DataConfig(**config_data)
    except Exception as e:
        # Re-raise our custom validation errors as-is
        if hasattr(e, '__class__') and 'ValidationError' in e.__class__.__name__:
            raise
        raise ValueError(f"Error creating DataConfig from {path}: {e}")


def validate_config_file(path: str) -> bool:
    """Validate a config file without loading it completely.
    
    Parameters
    ----------
    path : str
        Path to the configuration YAML file
        
    Returns
    -------
    bool
        True if valid, False otherwise
        
    Raises
    ------
    ConfigValidationError
        If validation fails (includes details about the issues)
    """
    try:
        read_config(path)
        return True
    except Exception:
        return False
