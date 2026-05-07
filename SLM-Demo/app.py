# app.py

import os
import pandas as pd
import streamlit as st
import plotly.express as px


RESULTS_FILE = "saved_demo_results.csv"


st.set_page_config(
    page_title="Saved SLM Demo Dashboard",
    page_icon="🤖",
    layout="wide"
)


@st.cache_data
def load_results(file_path):
    return pd.read_csv(file_path)


st.title("Small Language Model Demo Dashboard")
st.caption("Saved local Ollama demo results — no live model execution during presentation")


if not os.path.exists(RESULTS_FILE):
    st.error(f"Missing file: `{RESULTS_FILE}`")
    st.info("Run `python run_demo_once.py` first to generate saved demo results.")
    st.stop()


df = load_results(RESULTS_FILE)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Dashboard Controls")

models = sorted(df["model"].dropna().unique().tolist())
tasks = sorted(df["task_title"].dropna().unique().tolist())

selected_models = st.sidebar.multiselect(
    "Select models",
    models,
    default=models
)

selected_tasks = st.sidebar.multiselect(
    "Select demo tasks",
    tasks,
    default=tasks
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

if not selected_tasks:
    st.warning("Please select at least one task.")
    st.stop()


filtered_df = df[
    (df["model"].isin(selected_models)) &
    (df["task_title"].isin(selected_tasks))
]


# ---------------------------------------------------------
# Top explanation
# ---------------------------------------------------------

st.markdown("""
## Demo Idea

This dashboard shows a saved demo of **Small Language Models running locally**.

The models were asked to perform practical tasks:

- summarize messy text
- extract tasks and deadlines
- classify feedback
- generate structured JSON

The model outputs were generated before the presentation and saved into a CSV file.
So this dashboard is safe to present without running Ollama live.
""")


# ---------------------------------------------------------
# Main metrics
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Overall Demo Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Models Compared", filtered_df["model"].nunique())
col2.metric("Tasks Tested", filtered_df["task_title"].nunique())
col3.metric("Total Saved Runs", len(filtered_df))

success_rate = filtered_df["success"].mean() * 100
col4.metric("Success Rate", f"{success_rate:.1f}%")


# ---------------------------------------------------------
# Model comparison summary
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Model Comparison Summary")

summary_df = (
    filtered_df
    .groupby("model")
    .agg(
        avg_response_time_sec=("response_time_sec", "mean"),
        avg_words_per_second=("words_per_second", "mean"),
        avg_tokens_per_second_est=("tokens_per_second_est", "mean"),
        avg_output_words=("output_words", "mean"),
        avg_keyword_score_percent=("keyword_score_percent", "mean"),
        valid_json_rate=("valid_json", "mean"),
        success_rate=("success", "mean")
    )
    .reset_index()
)

summary_df["avg_response_time_sec"] = summary_df["avg_response_time_sec"].round(2)
summary_df["avg_words_per_second"] = summary_df["avg_words_per_second"].round(2)
summary_df["avg_tokens_per_second_est"] = summary_df["avg_tokens_per_second_est"].round(2)
summary_df["avg_output_words"] = summary_df["avg_output_words"].round(1)
summary_df["avg_keyword_score_percent"] = summary_df["avg_keyword_score_percent"].round(1)
summary_df["valid_json_rate"] = (summary_df["valid_json_rate"] * 100).round(1)
summary_df["success_rate"] = (summary_df["success_rate"] * 100).round(1)

display_summary = summary_df.rename(columns={
    "model": "Model",
    "avg_response_time_sec": "Avg Response Time (sec)",
    "avg_words_per_second": "Avg Words/sec",
    "avg_tokens_per_second_est": "Avg Tokens/sec Estimate",
    "avg_output_words": "Avg Output Words",
    "avg_keyword_score_percent": "Avg Keyword Score (%)",
    "valid_json_rate": "Valid JSON Rate (%)",
    "success_rate": "Success Rate (%)"
})

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Quick winner cards
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Quick Comparison")

if not summary_df.empty:
    fastest = summary_df.loc[summary_df["avg_response_time_sec"].idxmin()]
    highest_speed = summary_df.loc[summary_df["avg_words_per_second"].idxmax()]
    best_keyword = summary_df.loc[summary_df["avg_keyword_score_percent"].idxmax()]
    best_json = summary_df.loc[summary_df["valid_json_rate"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Fastest Model",
        fastest["model"],
        f'{fastest["avg_response_time_sec"]:.2f} sec avg'
    )

    c2.metric(
        "Highest Output Speed",
        highest_speed["model"],
        f'{highest_speed["avg_words_per_second"]:.2f} words/sec'
    )

    c3.metric(
        "Best Keyword Score",
        best_keyword["model"],
        f'{best_keyword["avg_keyword_score_percent"]:.1f}%'
    )

    c4.metric(
        "Best JSON Reliability",
        best_json["model"],
        f'{best_json["valid_json_rate"]:.1f}%'
    )


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Comparison Charts")

chart1, chart2 = st.columns(2)

with chart1:
    fig_time = px.bar(
        summary_df,
        x="model",
        y="avg_response_time_sec",
        title="Average Response Time by Model",
        labels={
            "model": "Model",
            "avg_response_time_sec": "Response Time (sec)"
        }
    )
    st.plotly_chart(fig_time, use_container_width=True)

with chart2:
    fig_speed = px.bar(
        summary_df,
        x="model",
        y="avg_words_per_second",
        title="Average Output Speed by Model",
        labels={
            "model": "Model",
            "avg_words_per_second": "Words per Second"
        }
    )
    st.plotly_chart(fig_speed, use_container_width=True)


chart3, chart4 = st.columns(2)

with chart3:
    fig_quality = px.bar(
        summary_df,
        x="model",
        y="avg_keyword_score_percent",
        title="Average Keyword Score by Model",
        labels={
            "model": "Model",
            "avg_keyword_score_percent": "Keyword Score (%)"
        }
    )
    st.plotly_chart(fig_quality, use_container_width=True)

with chart4:
    fig_json = px.bar(
        summary_df,
        x="model",
        y="valid_json_rate",
        title="Valid JSON Rate by Model",
        labels={
            "model": "Model",
            "valid_json_rate": "Valid JSON Rate (%)"
        }
    )
    st.plotly_chart(fig_json, use_container_width=True)


# ---------------------------------------------------------
# Task-level comparison
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Task-Level Model Comparison")

metric_choice = st.selectbox(
    "Choose comparison metric",
    [
        "response_time_sec",
        "words_per_second",
        "tokens_per_second_est",
        "keyword_score_percent",
        "output_words"
    ],
    format_func=lambda x: {
        "response_time_sec": "Response Time",
        "words_per_second": "Words per Second",
        "tokens_per_second_est": "Estimated Tokens per Second",
        "keyword_score_percent": "Keyword Score",
        "output_words": "Output Words"
    }[x]
)

fig_task = px.bar(
    filtered_df,
    x="task_title",
    y=metric_choice,
    color="model",
    barmode="group",
    title="Model Comparison by Task",
    labels={
        "task_title": "Task",
        "model": "Model",
        metric_choice: metric_choice.replace("_", " ").title()
    }
)

st.plotly_chart(fig_task, use_container_width=True)


# ---------------------------------------------------------
# Response viewer
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Saved Model Output Viewer")

viewer_col1, viewer_col2 = st.columns(2)

with viewer_col1:
    selected_model = st.selectbox(
        "Select model output to inspect",
        selected_models
    )

with viewer_col2:
    available_tasks = filtered_df[
        filtered_df["model"] == selected_model
    ]["task_title"].unique().tolist()

    selected_task = st.selectbox(
        "Select task output to inspect",
        available_tasks
    )


selected_row_df = filtered_df[
    (filtered_df["model"] == selected_model) &
    (filtered_df["task_title"] == selected_task)
]

if not selected_row_df.empty:
    row = selected_row_df.iloc[0]

    st.markdown("### Prompt Sent to Model")
    st.code(row["prompt"], language="text")

    st.markdown("### Saved Model Response")
    st.write(row["response"])

    st.markdown("### Metrics for This Output")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Response Time", f'{row["response_time_sec"]:.2f} sec')
    m2.metric("Words/sec", f'{row["words_per_second"]:.2f}')
    m3.metric("Est. Tokens/sec", f'{row["tokens_per_second_est"]:.2f}')
    m4.metric("Keyword Score", f'{row["keyword_score_percent"]:.1f}%')
    m5.metric("Valid JSON", str(row["valid_json"]))


# ---------------------------------------------------------
# Raw data
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Raw Saved Demo Results")

with st.expander("Show saved CSV data"):
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# Presentation talking points
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Presentation Talking Points")

st.markdown("""
### What this demo proves

This demo shows that Small Language Models can perform useful tasks locally:

- understand messy human text
- summarize information
- extract tasks and deadlines
- classify feedback
- return structured JSON

### Why saved results are used

The model was already run before the presentation.  
The outputs were saved to a CSV file and loaded into this dashboard.

This avoids problems during the presentation such as:

- slow CPU inference
- model loading delay
- terminal errors
- internet or system issues

### Main takeaway

Small Language Models are not always replacements for large models, but they are useful for:

- local AI
- low-cost automation
- privacy-focused workflows
- simple business tasks
- lightweight agent pipelines
""")


st.markdown("---")
st.caption("Saved SLM Demo Dashboard | Local Ollama | No live inference during presentation")