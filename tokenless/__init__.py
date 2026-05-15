"""
Tokenless: Use Kaggle's free GPU notebooks as an LLM inference backend.
"""

from tokenless.client import TokenlessLLM
from tokenless.notebook import GPT_OSS_MODEL_ID, run_gpt_oss_20b_prompt_on_kaggle
from tokenless.providers.openai_agents import TokenlessAgentsModel
from tokenless.providers.strands import TokenlessStrandsModel
from tokenless.providers.langchain import TokenlessLangChainLLM

__version__ = "0.1.0"
__all__ = [
    "TokenlessLLM",
    "TokenlessAgentsModel",
    "TokenlessStrandsModel",
    "TokenlessLangChainLLM",
    "GPT_OSS_MODEL_ID",
    "run_gpt_oss_20b_prompt_on_kaggle",
]
