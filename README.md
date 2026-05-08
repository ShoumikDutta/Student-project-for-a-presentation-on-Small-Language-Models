````markdown
# Student Project: Small Language Models

This repository contains a student project for a presentation on **Small Language Models (SLMs)**.  
The project demonstrates how smaller language models can be run locally using **Ollama**, tested with different prompt scenarios, benchmarked for speed and quality, and visualized through Streamlit dashboards.

## Project Goal

The goal of this project is to explore the practical strengths and limitations of Small Language Models.

The project focuses on:

- Running SLMs locally on a laptop
- Comparing multiple local models
- Measuring response time, token speed, and basic output quality
- Demonstrating common SLM use cases
- Showing limitations such as hallucination, weak reasoning, and structured output errors
- Creating simple dashboards for presentation purposes

## Repository Structure

```text
Student-project-for-a-presentation-on-Small-Language-Models/
│
├── Benchmark/
│   ├── benchmark_slm_ollama.py
│   ├── app.py
│   ├── benchmark_results_local.csv
│   ├── benchmark_results_local.json
│   ├── presentation_summary_local.csv
│   ├── scenario_summary_local.csv
│   ├── benchmark_results_groq.csv
│   ├── benchmark_results_groq.json
│   ├── presentation_summary_groq.csv
│   └── test_groq.py
│
├── SLM-Demo/
│   ├── run_demo_once.py
│   ├── slm_limitations_live_demo.py
│   ├── prompts.py
│   ├── app.py
│   ├── saved_demo_results.csv
│   └── saved_demo_results.json
│
└── .gitignore
````

## Main Parts of the Project

### 1. Benchmark

The `Benchmark` folder contains scripts for testing different Small Language Models locally through Ollama.

The benchmark tests models on different scenarios such as:

* Simple explanation
* Math reasoning
* Coding
* Summarization
* JSON structured output
* Factuality and hallucination check
* Instruction following
* RAG-style context answering

The benchmark records metrics such as:

* Response time
* Output tokens
* Tokens per second
* RAM change
* Keyword-based quality score
* JSON validity
* Instruction-following behavior

### 2. SLM Demo

The `SLM-Demo` folder contains a smaller demo designed for presentation.

It compares local models on practical tasks such as:

* Summarization
* Task extraction
* Structured JSON output
* Feedback classification

The demo saves the model outputs into CSV and JSON files so the results can be shown later without running the models live during the presentation.

### 3. Dashboards

Both folders include a Streamlit `app.py` file.

The dashboards are used to present the saved benchmark and demo results visually.
This makes it easier to compare models during the presentation without waiting for live model responses.

## Requirements

Install Python dependencies:

```bash
pip install requests pandas psutil streamlit plotly
```

You also need to install Ollama:

```bash
https://ollama.com/
```

## Ollama Setup

Start Ollama:

```bash
ollama serve
```

Pull the models used in the project:

```bash
ollama pull phi3
ollama pull phi3:mini
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull mistral
ollama pull tinyllama
ollama pull smollm2:1.7b
```

You do not need to use all models.
If a model is not installed, the benchmark script may skip it or show an error.

## How to Run the Benchmark

Go to the benchmark folder:

```bash
cd Benchmark
```

Run the local benchmark:

```bash
python benchmark_slm_ollama.py
```

This will generate or update result files such as:

```text
benchmark_results_local.csv
benchmark_results_local.json
presentation_summary_local.csv
scenario_summary_local.csv
```

## How to Open the Benchmark Dashboard

From inside the `Benchmark` folder, run:

```bash
streamlit run app.py
```

The dashboard shows:

* Models tested
* Scenarios tested
* Total benchmark runs
* Success rate
* Average response time
* Tokens per second
* Keyword score
* Scenario-level comparison

## How to Run the Demo

Go to the demo folder:

```bash
cd SLM-Demo
```

Run the saved demo script:

```bash
python run_demo_once.py
```

This will create or update:

```text
saved_demo_results.csv
saved_demo_results.json
```

## How to Open the Demo Dashboard

From inside the `SLM-Demo` folder, run:

```bash
streamlit run app.py
```

The dashboard can be used during the presentation to show saved model responses and compare the outputs.

## Example Demo Flow

The demo follows this basic flow:

```text
Prompt is selected
        ↓
Prompt is sent to a local Ollama model
        ↓
Model generates a response
        ↓
Script measures response time and output quality
        ↓
Result is saved to CSV/JSON
        ↓
Streamlit dashboard displays the results
```

## Models Used

The project mainly uses small or relatively lightweight open models through Ollama, such as:

* Gemma 3 1B
* Phi-3 Mini
* TinyLlama
* SmolLM2
* Mistral
* Gemma 3 4B

These models are useful for testing local AI because they can run without relying on cloud APIs.

## Why Small Language Models?

Small Language Models are useful because they can be:

* Faster than large models for simple tasks
* Cheaper to run
* Easier to deploy locally
* Better for privacy-sensitive use cases
* Suitable for laptops, edge devices, and small applications

However, they also have limitations:

* Weaker reasoning compared to large models
* More hallucination risk
* Less reliable structured output
* Lower performance on complex coding tasks
* Limited long-context understanding

## Notes for Presentation

This project is designed so that the model outputs can be generated before the presentation.
During the presentation, the saved dashboards can be shown instead of running the models live.

This avoids problems such as:

* Slow model loading
* Laptop performance issues
* Ollama connection errors
* Long response times during the presentation

## Important Security Note

Do not commit `.env` files or API keys to GitHub.

Recommended `.gitignore` entries:

```text
.env
Benchmark/.env
SLM-Demo/.env
__pycache__/
*.pyc
.venv/
```

If an API key was accidentally committed, delete or rotate the key immediately.

## Limitations of This Project

The benchmark uses simple keyword-based scoring, so it is not a perfect evaluation of model quality.
The results are useful for a student presentation and basic comparison, but not for professional model evaluation.

For more reliable evaluation, future improvements could include:

* Human scoring
* More test prompts
* Larger benchmark datasets
* Exact JSON schema validation
* Repeated runs for average performance
* Cost comparison between local and cloud models

## Author

Shoumik Dutta

## Project Context

This repository was created as part of a student presentation project on **Small Language Models**.

```
::contentReference[oaicite:3]{index=3}
```

[1]: https://github.com/ShoumikDutta/Student-project-for-a-presentation-on-Small-Language-Models "GitHub - ShoumikDutta/Student-project-for-a-presentation-on-Small-Language-Models · GitHub"
[2]: https://github.com/ShoumikDutta/Student-project-for-a-presentation-on-Small-Language-Models/tree/main/Benchmark "Student-project-for-a-presentation-on-Small-Language-Models/Benchmark at main · ShoumikDutta/Student-project-for-a-presentation-on-Small-Language-Models · GitHub"
[3]: https://github.com/ShoumikDutta/Student-project-for-a-presentation-on-Small-Language-Models/tree/main/SLM-Demo "Student-project-for-a-presentation-on-Small-Language-Models/SLM-Demo at main · ShoumikDutta/Student-project-for-a-presentation-on-Small-Language-Models · GitHub"
