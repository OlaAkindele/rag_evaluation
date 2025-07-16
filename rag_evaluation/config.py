# config.py

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from a .env file, if present.
# This enables API keys and other settings to be managed outside the codebase.
load_dotenv()

# internal cache
# A module-level dictionary used to store api keys for different providers
# during the current Python session. This avoids repeatedly accessing the environment
# or requiring re-entry of keys.
_api_key_cache: dict[str, str] = {}

def set_api_key(provider: str, key: str) -> None:
    """
    Store a provider's API key in an in-memory cache for the current Python session.

    parameters:
    provider : str
        The name of the API provider ("openai", "gemini").
    key : str
        The API key to associate with the provider.

    Raises
    ------
    ValueError
        If the provider name is empty.

    Example:
    import rag_evaluation as rag_eval
    rag_eval.set_api_key("openai", "sk-...")
    """
    if not provider:
        raise ValueError("provider cannot be empty")
    
    # Store the key under a normalized (lowercase) provider name
    _api_key_cache[provider.lower()] = key


def get_api_key(provider: str, default_key: Optional[str] = None) -> str:
    """
    Retrieve an API key for a given provider, using a prioritized fallback mechanism.

    Priority order:
        1. Use the key from the in-memory cache (if previously set via `set_api_key`)
        2. Look for an environment variable `${PROVIDER}_API_KEY`
        3. Use the provided `default_key` if given

    Parameters
    provider : str
        The name of the API provider to look up.
    default_key : Optional[str], optional
        A fallback key to return if no cached or environment key is found.

    Returns
    str
        The resolved API key.

    Raises
    ValueError
        If no API key is found through any of the available methods.

    Example:
    api_key = rag_eval.get_api_key("openai")
    """
    provider = provider.lower()

    # Check if the key was previously set in memory
    if provider in _api_key_cache:
        return _api_key_cache[provider]

    # Attempt to load the key from the environment using a conventional variable name
    env_var = f"{provider.upper()}_API_KEY"
    if env_key := os.getenv(env_var):
        return env_key

    # Fallback to the default key, if one was supplied
    if default_key:
        return default_key

    # If none of the above sources yielded a key, raise an informative error
    raise ValueError(
        f"{provider.capitalize()} API key not found. "
        f"Set it with rag_evaluation.set_api_key('{provider}', 'YOUR_API_KEY') "
        f"or export {env_var}=YOUR_API_KEY."
    )
