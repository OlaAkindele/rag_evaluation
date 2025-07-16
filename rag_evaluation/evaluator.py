import pandas as pd
from typing import Dict, Tuple, Any, Optional
import openai
from google import genai
from google.genai import types
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
        raise RuntimeError(f"OpenAI or Ollama evaluation API call failed: {e}")



def evaluate_gemini(criteria: str, steps: str, query: str, document: str,
                    response: str, metric_name: str, client, model: str) -> int:
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria, steps=steps, metric_name=metric_name,
        query=query, document=document, response=response
    )

    # build the new GenerateContentConfig
    gen_cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=16300,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0 # no thinking budget, just generate
        )
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
            score = evaluate_gemini(criteria, steps, query, document, response,
                                    metric_name, client, model_name)
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

    #use user-defined weights
    if "weights" in kwargs:
        # [Query Relevance, Factual Accuracy, Coverage, Coherence, Fluency]
        if len(kwargs["weights"]) != 5 or sum(kwargs["weights"]) != 1.0:
            raise ValueError(f"Must be a list of 5 values (Got {len(kwargs["weights"])}) or total of 1.0 (Got {sum(kwargs["weights"])}).")
        weights = kwargs["weights"]
    else:
        # default weights for the metrics
        weights = [0.25, 0.25, 0.25, 0.125, 0.125]

    if model_type.lower() == "openai" and "openai_client" not in kwargs:
        openai.api_key = get_api_key("openai")
        kwargs["openai_client"] = openai

    elif model_type.lower() == "ollama" and "ollama_client" not in kwargs:
        kwargs["ollama_client"] = openai.OpenAI(
            base_url='http://localhost:11434/v1', 
            api_key="ollama"  #works for advanced llama, qwen, and mistral models
        )
    elif model_type.lower() == "gemini" and "gemini_client" not in kwargs:
        gem_key = get_api_key("gemini")           # picks env/.env/cache as before
        kwargs["gemini_client"] = genai.Client(api_key=gem_key)

    eval_df = evaluate_all(evaluation_metrics, query, 
                           response, document, model_type, model_name, **kwargs)
    metric_names = list(evaluation_metrics.keys())
    scores = [eval_df.loc[metric, "Score"] for metric in metric_names]

    return generate_report(scores, weights, metric_names)
