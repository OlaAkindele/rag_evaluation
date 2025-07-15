import pandas as pd
from typing import Dict, Tuple, Any, Optional
import openai
import torch
from google import genai
from google.genai import types
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import get_api_key
from .utils import (
    normalize_score, calculate_weighted_accuracy,
    generate_report
)
from .metrics import (
    QUERY_RELEVANCE_CRITERIA, QUERY_RELEVANCE_STEPS,
    FACTUAL_ACCURACY_CRITERIA, FACTUAL_ACCURACY_STEPS,
    COVERAGE_CRITERIA, COVERAGE_STEPS, EVALUATION_PROMPT_TEMPLATE,
    COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS,
    FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS,
)


class LlamaEvaluator:    
    def __init__(self):
        self.llama_model = None
        self.llama_tokenizer = None
    
    def setup_llama(self, model_name: str, hf_token: Optional[str] = None):
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=hf_token
        )
        # Add padding token if it doesn't exist
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

def evaluate_llama(criteria: str, steps: str, query: str, document: str, response: str,
                   metric_name: str, model: Any, tokenizer: Any) -> int:

    raise NotImplementedError('Not Implemented')

def evaluate_openai(criteria: str, steps: str, query: str, document: str, response: str,
                    metric_name: str, client: Any, model: str) -> int:
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        query=query,
        document=document,
        response=response,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8192,
        )
        # print(resp.choices[0].message.content)
        return int(resp.choices[0].message.content.strip())
    except Exception as e:
        raise RuntimeError(f"OpenAI evaluation API call failed: {e}")



def evaluate_gemini(criteria: str, steps: str, query: str, document: str,
                    response: str, metric_name: str, client, model: str) -> int:
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria, steps=steps, metric_name=metric_name,
        query=query, document=document, response=response
    )

    # build the new GenerateContentConfig
    gen_cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=16_300,
    )
    try:
        # NEW SDK signature (>= 1.0)
        gem_resp = client.models.generate_content(
            model=model,
            contents=prompt,        # a plain str is fine; SDK converts it
            config=gen_cfg,
        )
    except TypeError:
        # fallback for very old (< 0.8) SDKs that still want generation_config
        gem_resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            generation_config=types.GenerationConfig(**gen_cfg.model_dump()),
        )

    raw = getattr(gem_resp, "text", None) \
          or (gem_resp.candidates[0].content if gem_resp.candidates else None)
    if not raw:
        raise RuntimeError(f"No content returned by Gemini (got: {gem_resp!r})")

    return int(raw.replace("\n", "").strip())



def evaluate_all(metrics: Dict[str, Tuple[str, str]], query: str, response: str,
                 document: str, model_type: str, model_name: str, **kwargs) -> pd.DataFrame:
    data = {"Evaluation Type": [], "Score": []}

    for metric_name, (criteria, steps) in metrics.items():
        data["Evaluation Type"].append(metric_name)

        if model_type.lower() == "openai":
            client = kwargs.get("openai_client")
            score = evaluate_openai(criteria, steps, query, document, response,
                                    metric_name, client, model_name)
        elif model_type.lower() == "ollama":
            client = kwargs.get("ollama_client")
            score = evaluate_openai(criteria, steps, query, document, response,
                                    metric_name, client, model_name)
        elif model_type.lower() == "gemini":
            client = kwargs.get("gemini_client")
            if not client:
                raise ValueError("Gemini client not provided.")
            score = evaluate_gemini(criteria, steps, query, document, response,
                                    metric_name, client, model_name)
        # elif model_type.lower() == "llama":
        #     model = kwargs.get("llama_model")
        #     tokenizer = kwargs.get("llama_tokenizer")
        #     if not model or not tokenizer:
        #         raise ValueError("Llama model/tokenizer not provided.")
        #     score = evaluate_llama(criteria, steps, query, document, response,
        #                            metric_name, model, tokenizer)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        data["Score"].append(score)

    return pd.DataFrame(data).set_index("Evaluation Type")


def evaluate_response(query: str, response: str, document: str,
                      model_type: str, model_name: str, **kwargs) -> pd.DataFrame:
    evaluation_metrics = {
        "Query Relevance": (QUERY_RELEVANCE_CRITERIA, QUERY_RELEVANCE_STEPS),
        "Factual Accuracy": (FACTUAL_ACCURACY_CRITERIA, FACTUAL_ACCURACY_STEPS),
        "Coverage": (COVERAGE_CRITERIA, COVERAGE_STEPS),
        "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
        "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
    }
    default_weights = [0.25, 0.25, 0.25, 0.125, 0.125]  # [0.25, 0.25, 0.25, 0.125, 0.125]  = 1.00

    # if model_type.lower() == "openai" and "openai_client" not in kwargs:
    #     kwargs["openai_client"] = openai
    if model_type.lower() == "openai" and "openai_client" not in kwargs:
        openai.api_key = get_api_key("openai")
        kwargs["openai_client"] = openai

    elif model_type.lower() == "ollama" and "ollama_client" not in kwargs:
        kwargs["ollama_client"] = openai.OpenAI(
            base_url='http://localhost:11434/v1', 
            api_key="ollama"  #works for llama, quewen, and mistral models
        )
    # elif model_type.lower() == "gemini" and "gemini_client" not in kwargs:
    #     kwargs["gemini_client"] = genai.Client()
    elif model_type.lower() == "gemini" and "gemini_client" not in kwargs:
        gem_key = get_api_key("gemini")           # picks env/.env/cache as before
        kwargs["gemini_client"] = genai.Client(api_key=gem_key)

    eval_df = evaluate_all(evaluation_metrics, query, 
                           response, document, model_type, model_name, **kwargs)
    metric_names = list(evaluation_metrics.keys())
    scores = [eval_df.loc[metric, "Score"] for metric in metric_names]

    return generate_report(scores, default_weights, metric_names)
