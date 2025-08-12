import yaml
import os
from dotenv import load_dotenv

def load_config(path: str = "config.yaml") -> dict:
    """
    Loads configuration from a YAML file and merges it with environment variables.

    Args:
        path (str): The path to the configuration YAML file.

    Returns:
        dict: A dictionary containing the loaded configuration.
    """
    # Load environment variables from a .env file if it exists
    load_dotenv()

    # Load base configuration from YAML file
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Merge environment variables into the config
    # This allows overriding YAML settings with environment variables
    for key, value in os.environ.items():
        # Simple override for top-level keys, can be expanded for nested keys
        if key in config:
            config[key] = value

    # You can also specifically look for keys, e.g., API keys
    if 'OPENAI_API_KEY' in os.environ:
        if 'api_keys' not in config:
            config['api_keys'] = {}
        config['api_keys']['openai'] = os.environ['OPENAI_API_KEY']

    return config
