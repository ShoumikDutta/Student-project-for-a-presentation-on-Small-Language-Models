# app.py

import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="SLM Benchmark Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("Small Language Model Benchmark Dashboard")
st.caption("Local Ollama benchmark results for Small Language Models")


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BENCHMARK_FILE = BASE_DIR / "benchmark_results_local.csv"  #pd.read_csv("Benchmark/benchmark_results_local.csv")
PRESENTATION_FILE = pd.read_csv("Benchmark/presentation_summary_local.csv")
SCENARIO_FILE = pd.read_csv("Benchmark/scenario_summary_local.csv")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def load_csv(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)


def clean_model_name(name):
    return str(name).replace(":", "_")


def show_missing_file_warning(file_name):
    st.error(f"Missing file: `{file_name}`")
    st.info("Make sure the CSV file is in the same folder as `app.py`.")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

benchmark_df = load_csv(BENCHMARK_FILE)
presentation_df = load_csv(PRESENTATION_FILE)
scenario_df = load_csv(SCENARIO_FILE)

if benchmark_df is None:
    show_missing_file_warning(BENCHMARK_FILE)
    st.stop()

if presentation_df is None:
    show_missing_file_warning(PRESENTATION_FILE)
    st.stop()

if scenario_df is None:
    show_missing_file_warning(SCENARIO_FILE)
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Dashboard Filters")

all_models = sorted(benchmark_df["model"].dropna().unique().tolist())

selected_models = st.sidebar.multiselect(
    "Select models",
    all_models,
    default=all_models
)

all_scenarios = sorted(benchmark_df["scenario"].dropna().unique().tolist())

selected_scenarios = st.sidebar.multiselect(
    "Select scenarios",
    all_scenarios,
    default=all_scenarios
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

if not selected_scenarios:
    st.warning("Please select at least one scenario.")
    st.stop()


filtered_benchmark = benchmark_df[
    (benchmark_df["model"].isin(selected_models)) &
    (benchmark_df["scenario"].isin(selected_scenarios))
]

filtered_presentation = presentation_df[
    presentation_df["model"].isin(selected_models)
]

filtered_scenario = scenario_df[
    (scenario_df["model"].isin(selected_models)) &
    (scenario_df["scenario"].isin(selected_scenarios))
]


# ---------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------

st.subheader("Benchmark Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Models Tested", filtered_benchmark["model"].nunique())
col2.metric("Scenarios Tested", filtered_benchmark["scenario"].nunique())
col3.metric("Total Runs", len(filtered_benchmark))

success_rate = filtered_benchmark["success"].mean() * 100 if "success" in filtered_benchmark.columns else 0
col4.metric("Success Rate", f"{success_rate:.1f}%")


# ---------------------------------------------------------
# Presentation summary
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Model Summary")

summary_display = filtered_presentation.copy()

summary_display = summary_display.rename(columns={
    "provider": "Provider",
    "model": "Model",
    "avg_response_time_sec": "Avg Response Time (sec)",
    "avg_tokens_per_second": "Avg Tokens/sec",
    "avg_output_tokens": "Avg Output Tokens",
    "avg_keyword_score": "Avg Keyword Score",
    "avg_ram_change_mb": "Avg RAM Change (MB)"
})

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Best model cards
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Quick Takeaways")

if not filtered_presentation.empty:
    fastest_model = filtered_presentation.loc[
        filtered_presentation["avg_response_time_sec"].idxmin()
    ]

    best_quality_model = filtered_presentation.loc[
        filtered_presentation["avg_keyword_score"].idxmax()
    ]

    best_speed_model = filtered_presentation.loc[
        filtered_presentation["avg_tokens_per_second"].idxmax()
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fastest Model",
        fastest_model["model"],
        f'{fastest_model["avg_response_time_sec"]:.2f} sec'
    )

    c2.metric(
        "Highest Tokens/sec",
        best_speed_model["model"],
        f'{best_speed_model["avg_tokens_per_second"]:.2f}'
    )

    c3.metric(
        "Best Keyword Score",
        best_quality_model["model"],
        f'{best_quality_model["avg_keyword_score"]:.2f}'
    )


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Performance Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_time = px.bar(
        filtered_presentation,
        x="model",
        y="avg_response_time_sec",
        title="Average Response Time by Model",
        labels={
            "model": "Model",
            "avg_response_time_sec": "Average Response Time (sec)"
        }
    )
    st.plotly_chart(fig_time, use_container_width=True)

with chart_col2:
    fig_speed = px.bar(
        filtered_presentation,
        x="model",
        y="avg_tokens_per_second",
        title="Average Tokens per Second by Model",
        labels={
            "model": "Model",
            "avg_tokens_per_second": "Tokens per Second"
        }
    )
    st.plotly_chart(fig_speed, use_container_width=True)


chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_quality = px.bar(
        filtered_presentation,
        x="model",
        y="avg_keyword_score",
        title="Average Keyword Score by Model",
        labels={
            "model": "Model",
            "avg_keyword_score": "Keyword Score"
        }
    )
    st.plotly_chart(fig_quality, use_container_width=True)

with chart_col4:
    fig_ram = px.bar(
        filtered_presentation,
        x="model",
        y="avg_ram_change_mb",
        title="Average RAM Change by Model",
        labels={
            "model": "Model",
            "avg_ram_change_mb": "RAM Change (MB)"
        }
    )
    st.plotly_chart(fig_ram, use_container_width=True)


# ---------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Scenario-Level Comparison")

scenario_metric = st.selectbox(
    "Choose metric for scenario comparison",
    [
        "response_time_sec",
        "tokens_per_second",
        "keyword_score",
        "output_tokens"
    ],
    format_func=lambda x: {
        "response_time_sec": "Response Time",
        "tokens_per_second": "Tokens per Second",
        "keyword_score": "Keyword Score",
        "output_tokens": "Output Tokens"
    }[x]
)

fig_scenario = px.bar(
    filtered_scenario,
    x="scenario",
    y=scenario_metric,
    color="model",
    barmode="group",
    title="Model Performance by Scenario",
    labels={
        "scenario": "Scenario",
        "model": "Model",
        scenario_metric: scenario_metric.replace("_", " ").title()
    }
)

st.plotly_chart(fig_scenario, use_container_width=True)


# ---------------------------------------------------------
# Scatter plot: speed vs quality
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Speed vs Quality Trade-off")

fig_scatter = px.scatter(
    filtered_presentation,
    x="avg_response_time_sec",
    y="avg_keyword_score",
    size="avg_output_tokens",
    color="model",
    hover_name="model",
    title="Response Time vs Keyword Score",
    labels={
        "avg_response_time_sec": "Average Response Time (sec)",
        "avg_keyword_score": "Average Keyword Score",
        "avg_output_tokens": "Average Output Tokens"
    }
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.info(
    "Lower response time is better. Higher keyword score is better. "
    "This chart shows the trade-off between speed and output quality."
)


# ---------------------------------------------------------
# Raw benchmark results
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Detailed Benchmark Results")

columns_to_show = [
    "timestamp",
    "provider",
    "model",
    "scenario",
    "success",
    "wall_time_sec",
    "tokens_per_second",
    "output_tokens",
    "ram_change_mb",
    "keyword_score",
    "valid_json",
    "response",
    "error"
]

available_columns = [
    col for col in columns_to_show if col in filtered_benchmark.columns
]

st.dataframe(
    filtered_benchmark[available_columns],
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Model response viewer
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Model Response Viewer")

selected_model_response = st.selectbox(
    "Select model to inspect",
    selected_models
)

available_response_scenarios = filtered_benchmark[
    filtered_benchmark["model"] == selected_model_response
]["scenario"].dropna().unique().tolist()

selected_response_scenario = st.selectbox(
    "Select scenario to inspect",
    available_response_scenarios
)

response_rows = filtered_benchmark[
    (filtered_benchmark["model"] == selected_model_response) &
    (filtered_benchmark["scenario"] == selected_response_scenario)
]

if not response_rows.empty:
    row = response_rows.iloc[0]

    st.markdown("### Prompt")
    st.code(row.get("prompt", "No prompt available"), language="text")

    st.markdown("### Model Response")
    st.write(row.get("response", "No response available"))

    st.markdown("### Metrics for This Run")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Wall Time", f'{row.get("wall_time_sec", 0):.2f} sec')
    m2.metric("Tokens/sec", f'{row.get("tokens_per_second", 0):.2f}')
    m3.metric("Output Tokens", int(row.get("output_tokens", 0)))
    m4.metric("Keyword Score", f'{row.get("keyword_score", 0):.2f}')


# ---------------------------------------------------------
# Presentation talking points
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Presentation Talking Points")

st.markdown("""
### What this dashboard shows

This dashboard compares small language models running locally with Ollama.

The benchmark measures:

- **Response time** — how long the model takes to answer
- **Tokens per second** — how fast the model generates text
- **Output tokens** — how long the answer is
- **Keyword score** — simple quality check based on expected keywords
- **RAM change** — approximate memory impact during the run

### Main message

Small Language Models are useful when we need:

- local execution
- lower cost
- privacy
- lightweight automation
- simple structured tasks

They may not always beat large models in reasoning quality, but they are useful for smaller workflows like summarization, classification, task extraction, and JSON generation.
""")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")
st.caption("SLM Benchmark Dashboard | Local Ollama | Student Presentation Project")
