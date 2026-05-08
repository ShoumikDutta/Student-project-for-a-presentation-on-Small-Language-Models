# run_demo_once.py

import csv
import json
import re
import time
from datetime import datetime

import requests

from prompts import PROMPTS


# ---------------------------------------------------------
# Models to compare
# ---------------------------------------------------------
# Change these if needed.
# Make sure the models are already pulled with:
# ollama pull gemma3:1b
# ollama pull phi3:mini

MODELS = [
    "gemma3:1b",
    "phi3:mini"
]


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

OUTPUT_CSV = "saved_demo_results.csv"
OUTPUT_JSON = "saved_demo_results.json"


# ---------------------------------------------------------
# Ollama API settings
# ---------------------------------------------------------

OLLAMA_API_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def run_ollama(model: str, prompt: str, timeout: int = 240) -> dict:
    """
    Runs a local Ollama model using the Ollama HTTP API.

    This is cleaner than using:
        subprocess.run(["ollama", "run", model])

    because the API response does not include terminal control characters.
    """

    start_time = time.time()

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=timeout
        )

        duration = time.time() - start_time

        if response.status_code != 200:
            return {
                "success": False,
                "response": "",
                "error": response.text,
                "response_time_sec": round(duration, 2)
            }

        data = response.json()

        return {
            "success": True,
            "response": data.get("response", "").strip(),
            "error": "",
            "response_time_sec": round(duration, 2)
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "response": "",
            "error": (
                "Could not connect to Ollama. "
                "Make sure Ollama is running and open http://localhost:11434 in browser."
            ),
            "response_time_sec": 0
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "response": "",
            "error": "Timeout: model took too long to respond.",
            "response_time_sec": timeout
        }

    except Exception as e:
        return {
            "success": False,
            "response": "",
            "error": str(e),
            "response_time_sec": 0
        }


def clean_response_text(text: str) -> str:
    """
    Cleans model output.

    Removes:
    - ANSI terminal escape characters
    - Markdown JSON code fences
    - Extra whitespace
    """

    if not isinstance(text, str):
        return ""

    # Remove ANSI escape sequences like: \x1b[3D\x1b[K
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

    # Remove markdown code fences
    text = text.replace("```json", "")
    text = text.replace("```JSON", "")
    text = text.replace("```", "")

    return text.strip()


def extract_json_part(text: str) -> str:
    """
    Extracts JSON object from the model response.

    Example:
        Here is the JSON:
        {
          "tasks": []
        }

    It extracts only:
        {
          "tasks": []
        }
    """

    cleaned = clean_response_text(text)

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1

    if start == -1 or end == 0:
        return ""

    return cleaned[start:end].strip()


def is_valid_json(text: str) -> bool:
    """
    Checks whether the response contains valid JSON.
    """

    json_part = extract_json_part(text)

    if not json_part:
        return False

    try:
        json.loads(json_part)
        return True
    except Exception:
        return False


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate.

    This is not exact tokenization.
    For a simple presentation metric, this is enough.
    """

    if not isinstance(text, str):
        return 0

    words = text.split()
    return int(len(words) * 1.3)


def keyword_score(response: str, expected_keywords: list[str]) -> float:
    """
    Simple quality score.

    It checks how many expected keywords appear in the response.
    """

    if not expected_keywords:
        return 0.0

    response_lower = response.lower()
    matched = 0

    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            matched += 1

    return round((matched / len(expected_keywords)) * 100, 2)


def safe_print(text: str):
    """
    Prints safely in terminal.
    """

    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="ignore").decode("utf-8"))


# ---------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------

def main():
    results = []

    print("=" * 80)
    print("Running Saved Small Language Model Demo")
    print("=" * 80)
    print("This script runs local Ollama models once and saves outputs to CSV/JSON.")
    print("During presentation, use the saved dashboard instead of running models live.")
    print("=" * 80)

    for model in MODELS:
        for task_key, task_data in PROMPTS.items():
            print("\n" + "-" * 80)
            print(f"Model: {model}")
            print(f"Task: {task_data['title']}")
            print("-" * 80)

            prompt = task_data["prompt"]

            run_result = run_ollama(
                model=model,
                prompt=prompt
            )

            raw_response = run_result["response"]
            clean_response = clean_response_text(raw_response)
            extracted_json = extract_json_part(raw_response)
            valid_json = is_valid_json(raw_response)

            response_time = run_result["response_time_sec"]
            output_words = len(clean_response.split())
            output_tokens_est = estimate_tokens(clean_response)

            words_per_second = (
                round(output_words / response_time, 2)
                if response_time > 0
                else 0
            )

            tokens_per_second_est = (
                round(output_tokens_est / response_time, 2)
                if response_time > 0
                else 0
            )

            score = keyword_score(
                response=clean_response,
                expected_keywords=task_data.get("expected_keywords", [])
            )

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "task_key": task_key,
                "task_title": task_data["title"],
                "prompt": prompt.strip(),
                "response": clean_response,
                "raw_response": raw_response,
                "clean_response": clean_response,
                "extracted_json": extracted_json,
                "success": run_result["success"],
                "error": run_result["error"],
                "response_time_sec": response_time,
                "output_words": output_words,
                "output_tokens_est": output_tokens_est,
                "words_per_second": words_per_second,
                "tokens_per_second_est": tokens_per_second_est,
                "valid_json": valid_json,
                "keyword_score_percent": score
            }

            results.append(row)

            print(f"Success: {row['success']}")
            print(f"Response time: {row['response_time_sec']} sec")
            print(f"Output words: {row['output_words']}")
            print(f"Words/sec: {row['words_per_second']}")
            print(f"Estimated tokens/sec: {row['tokens_per_second_est']}")
            print(f"Valid JSON: {row['valid_json']}")
            print(f"Keyword score: {row['keyword_score_percent']}%")

            if row["error"]:
                print("Error:")
                safe_print(row["error"])

    if not results:
        print("No results created.")
        return

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    fieldnames = list(results[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    with open(OUTPUT_JSON, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Demo saved successfully.")
    print(f"CSV file created:  {OUTPUT_CSV}")
    print(f"JSON file created: {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()