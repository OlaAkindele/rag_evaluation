import os
from dotenv import load_dotenv

# API key management and configuration

# Load environment variables from a .env file (if available)
load_dotenv()

import os

def get_api_key(provider: str, default_key: str = None) -> str:
    """
    Retrieve the API key for a specified provider from the environment or a provided default.
    
    Parameters:
        provider (str): The provider name ("openai", "gemini").
        default_key (str): A fallback API key if the environment variable is not set.
    
    Returns:
        str: The API key.
    
    Raises:
        ValueError: If no API key is found.
    """
    env_var = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var, default_key)
    if not api_key:
        raise ValueError(f"{provider.capitalize()} API key is not set. Please set the '{env_var}' environment variable or provide a default.")
    return api_key

