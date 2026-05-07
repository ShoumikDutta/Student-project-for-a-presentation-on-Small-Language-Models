import requests
import time
import json
import csv
import psutil
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


# ============================================================
# LOCAL OLLAMA BENCHMARK FOR SMALL LANGUAGE MODELS
# ============================================================
# This script runs locally through Ollama:
#   http://localhost:11434/api/generate
#
# Install:
#   pip install requests psutil pandas
#
# Before running:
#   ollama serve
#
# Pull models:
#   ollama pull phi3
#   ollama pull gemma:2b
#   ollama pull tinyllama
#   ollama pull smollm2:1.7b
#   ollama pull qwen3:4b
#
# Run:
#   python benchmark_slm_local.py
# ============================================================


OLLAMA_URL = "http://localhost:11434/api/generate"


MODELS = [
    "phi3",
    "gemma3:1b",
    "gemma3:4b",
    "mistral",
    "tinyllama",
    "smollm2:1.7b",
]

TEMPERATURE = 0.0


PROMPTS = [
    {
        "scenario": "simple_explanation",
        "prompt": "Explain what a KV cache is in transformers in 3 simple bullet points.",
        "expected_keywords": ["cache", "key", "value", "tokens", "attention"],
    },
    {
        "scenario": "reasoning_math",
        "prompt": (
            "A student buys 3 notebooks for 2.50 euros each and 2 pens for 1.20 euros each. "
            "They pay with a 10 euro note. How much change should they get? Show the calculation briefly."
        ),
        "expected_keywords": ["7.50", "2.40", "9.90", "0.10"],
    },
    {
        "scenario": "coding",
        "prompt": (
            "Write a Python function called is_prime(n) that returns True if n is prime "
            "and False otherwise. Keep the code short and readable."
        ),
        "expected_keywords": ["def", "is_prime", "return", "False", "True"],
    },
    {
        "scenario": "summarization",
        "prompt": (
            "Summarize the following text in exactly 3 bullet points:\n\n"
            "Small Language Models are compact neural networks designed to perform language tasks "
            "with fewer parameters than large frontier models. They are useful for local deployment, "
            "privacy-sensitive applications, and low-cost inference. However, they usually perform worse "
            "than large models on complex reasoning, long-context tasks, and advanced coding."
        ),
        "expected_keywords": ["local", "privacy", "cost", "reasoning"],
    },
    {
        "scenario": "json_structured_output",
        "prompt": (
            "Analyze this feedback and respond only in valid JSON with keys: sentiment, issue, urgency.\n\n"
            "Feedback: The app is useful, but it crashes every time I upload a large PDF."
        ),
        "expected_keywords": ["sentiment", "issue", "urgency"],
    },
    {
        "scenario": "factuality_hallucination_check",
        "prompt": (
            "If you do not know the answer, say 'I do not know'. "
            "Question: Who won the Nobel Prize in Physics in 2035?"
        ),
        "expected_keywords": ["do not know", "don't know", "cannot know", "future"],
    },
    {
        "scenario": "instruction_following",
        "prompt": (
            "Write exactly two sentences about why local AI can be useful. "
            "Do not use bullet points."
        ),
        "expected_keywords": ["local", "AI"],
    },
    {
        "scenario": "rag_style_context",
        "prompt": (
            "Use only the context below to answer the question. "
            "If the answer is not in the context, say you do not know.\n\n"
            "Context: The company uses Ollama to run small language models locally. "
            "The main benefit is that private documents do not leave the laptop. "
            "The tested model was Phi-3 Mini.\n\n"
            "Question: Which model was tested and what was the main privacy benefit?"
        ),
        "expected_keywords": ["Phi-3", "private", "laptop", "documents"],
    },
]


def check_ollama_running() -> bool:
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_installed_models() -> List[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def get_memory_usage_mb() -> float:
    memory = psutil.virtual_memory()
    return round(memory.used / (1024 * 1024), 2)


def call_ollama(model: str, prompt: str) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
        },
    }

    start_time = time.time()
    ram_before = get_memory_usage_mb()

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        end_time = time.time()
        ram_after = get_memory_usage_mb()

        wall_time = end_time - start_time
        output_text = data.get("response", "")

        # Ollama durations are returned in nanoseconds
        total_duration = data.get("total_duration", 0) / 1e9
        load_duration = data.get("load_duration", 0) / 1e9
        prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9
        eval_duration = data.get("eval_duration", 0) / 1e9

        prompt_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        tokens_per_second = None
        if eval_duration > 0 and output_tokens > 0:
            tokens_per_second = round(output_tokens / eval_duration, 2)

        return {
            "success": True,
            "provider": "Local Ollama",
            "response": output_text,
            "wall_time_sec": round(wall_time, 3),
            "ollama_total_duration_sec": round(total_duration, 3),
            "load_duration_sec": round(load_duration, 3),
            "prompt_eval_duration_sec": round(prompt_eval_duration, 3),
            "eval_duration_sec": round(eval_duration, 3),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "ram_before_mb": ram_before,
            "ram_after_mb": ram_after,
            "ram_change_mb": round(ram_after - ram_before, 2),
            "error": None,
        }

    except Exception as e:
        end_time = time.time()

        return {
            "success": False,
            "provider": "Local Ollama",
            "response": "",
            "wall_time_sec": round(end_time - start_time, 3),
            "ollama_total_duration_sec": None,
            "load_duration_sec": None,
            "prompt_eval_duration_sec": None,
            "eval_duration_sec": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "tokens_per_second": None,
            "ram_before_mb": None,
            "ram_after_mb": None,
            "ram_change_mb": None,
            "error": str(e),
        }


def keyword_score(response: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0

    response_lower = response.lower()
    matched = 0

    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            matched += 1

    return round(matched / len(expected_keywords), 2)


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def count_sentences(text: str) -> int:
    endings = [".", "!", "?"]
    return sum(1 for char in text if char in endings)


def has_bullet_points(text: str) -> bool:
    bullet_markers = ["- ", "* ", "• ", "1.", "2.", "3."]
    return any(marker in text for marker in bullet_markers)


def extra_checks(scenario: str, response: str) -> Dict[str, Any]:
    checks = {
        "valid_json": None,
        "sentence_count": None,
        "has_bullet_points": None,
    }

    if scenario == "json_structured_output":
        checks["valid_json"] = is_valid_json(response)

    if scenario == "instruction_following":
        checks["sentence_count"] = count_sentences(response)
        checks["has_bullet_points"] = has_bullet_points(response)

    return checks


def run_benchmark() -> List[Dict[str, Any]]:
    results = []

    print("=" * 90)
    print("SLM BENCHMARK USING LOCAL OLLAMA")
    print("This benchmark runs on your laptop through Ollama.")
    print("=" * 90)

    if not check_ollama_running():
        print("ERROR: Ollama is not running.")
        print("Start Ollama first with:")
        print("  ollama serve")
        return results

    installed_models = get_installed_models()

    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models selected: {MODELS}")
    print(f"Installed Ollama models: {installed_models}")
    print(f"Number of prompt scenarios: {len(PROMPTS)}")
    print("=" * 90)

    for model in MODELS:
        is_installed = any(
            installed == model or installed.startswith(model + ":")
            for installed in installed_models
        )

        if not is_installed:
            print(f"\nSkipping {model}: model not installed.")
            print(f"Pull it with: ollama pull {model}")
            continue

        print(f"\n\nTesting model: {model}")
        print("-" * 90)

        for item in PROMPTS:
            scenario = item["scenario"]
            prompt = item["prompt"]
            expected_keywords = item.get("expected_keywords", [])

            print(f"\nScenario: {scenario}")

            result = call_ollama(model, prompt)
            quality_score = keyword_score(result["response"], expected_keywords)
            checks = extra_checks(scenario, result["response"])

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "provider": result["provider"],
                "model": model,
                "scenario": scenario,
                "prompt": prompt,
                "success": result["success"],
                "wall_time_sec": result["wall_time_sec"],
                "ollama_total_duration_sec": result["ollama_total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],
                "prompt_eval_duration_sec": result["prompt_eval_duration_sec"],
                "eval_duration_sec": result["eval_duration_sec"],
                "prompt_tokens": result["prompt_tokens"],
                "output_tokens": result["output_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "ram_before_mb": result["ram_before_mb"],
                "ram_after_mb": result["ram_after_mb"],
                "ram_change_mb": result["ram_change_mb"],
                "keyword_score": quality_score,
                "valid_json": checks["valid_json"],
                "sentence_count": checks["sentence_count"],
                "has_bullet_points": checks["has_bullet_points"],
                "response": result["response"],
                "error": result["error"],
            }

            results.append(row)

            if result["success"]:
                print(
                    f"  Provider: {result['provider']} | "
                    f"Time: {result['wall_time_sec']} sec | "
                    f"Output tokens: {result['output_tokens']} | "
                    f"Speed: {result['tokens_per_second']} tokens/sec | "
                    f"Keyword score: {quality_score}"
                )

                preview = result["response"].replace("\n", " ")[:180]
                print(f"  Preview: {preview}...")

                if scenario == "json_structured_output":
                    print(f"  Valid JSON: {checks['valid_json']}")

                if scenario == "instruction_following":
                    print(
                        f"  Sentence count: {checks['sentence_count']} | "
                        f"Bullet points used: {checks['has_bullet_points']}"
                    )
            else:
                print(f"  ERROR: {result['error']}")

    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    csv_file = "benchmark_results_local.csv"
    json_file = "benchmark_results_local.json"

    if not results:
        print("No results to save.")
        return

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = list(results[0].keys())

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 90)
    print("Local Ollama benchmark completed.")
    print(f"Saved CSV:  {csv_file}")
    print(f"Saved JSON: {json_file}")
    print("=" * 90)


def print_summary(results: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(results)

    successful = df[df["success"] == True].copy()

    if successful.empty:
        print("No successful benchmark results.")
        return

    model_summary = successful.groupby(["provider", "model"]).agg(
        avg_response_time_sec=("wall_time_sec", "mean"),
        avg_tokens_per_second=("tokens_per_second", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_keyword_score=("keyword_score", "mean"),
        avg_ram_change_mb=("ram_change_mb", "mean"),
    ).reset_index()

    model_summary = model_summary.round(2)

    print("\n\nMODEL SUMMARY")
    print("=" * 90)
    print(model_summary.to_string(index=False))

    scenario_summary = successful.groupby(["model", "scenario"]).agg(
        response_time_sec=("wall_time_sec", "mean"),
        tokens_per_second=("tokens_per_second", "mean"),
        keyword_score=("keyword_score", "mean"),
        output_tokens=("output_tokens", "mean"),
    ).reset_index()

    scenario_summary = scenario_summary.round(2)

    print("\n\nSCENARIO SUMMARY")
    print("=" * 90)
    print(scenario_summary.to_string(index=False))

    model_summary.to_csv("presentation_summary_local.csv", index=False)
    scenario_summary.to_csv("scenario_summary_local.csv", index=False)

    print("\nSaved presentation summary: presentation_summary_local.csv")
    print("Saved scenario summary: scenario_summary_local.csv")


if __name__ == "__main__":
    results = run_benchmark()
    save_results(results)
    print_summary(results)