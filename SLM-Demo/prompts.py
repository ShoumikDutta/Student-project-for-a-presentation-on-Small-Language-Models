# prompts.py

DEMO_NOTE = """
Hi Shoumik, please remind Faisal that the lab report is due on 12 May.
Also email the professor to ask if we can join Lab Group 1.
Meeting with Nils is on Thursday at 14:00.
The dashboard loading is slow, so maybe we should test a smaller model.
"""

PROMPTS = {
    "summarization": {
        "title": "Summarization",
        "expected_keywords": ["lab report", "professor", "meeting", "dashboard"],
        "prompt": f"""
You are a small language model running locally.

Task:
Summarize the following note in one short sentence.

Note:
{DEMO_NOTE}
"""
    },

    "task_extraction": {
        "title": "Task Extraction",
        "expected_keywords": ["Faisal", "lab report", "12 May", "professor", "Nils"],
        "prompt": f"""
You are a small language model running locally.

Task:
Extract all tasks, people, dates, and actions from the note.

Return the answer in clear bullet points.

Note:
{DEMO_NOTE}
"""
    },

    "json_output": {
        "title": "Structured JSON Output",
        "expected_keywords": ["tasks", "person", "deadline", "issues", "dashboard"],
        "prompt": f"""
You are a small language model running locally.

Task:
Extract useful information from the note and return ONLY valid JSON.

Use this structure:
{{
  "tasks": [
    {{
      "task": "string",
      "person": "string or null",
      "deadline_or_time": "string or null",
      "suggested_action": "string"
    }}
  ],
  "issues": [
    {{
      "issue": "string",
      "suggested_solution": "string"
    }}
  ]
}}

Note:
{DEMO_NOTE}
"""
    },

    "classification": {
        "title": "Feedback Classification",
        "expected_keywords": ["mixed", "useful", "slow", "loading"],
        "prompt": """
You are a small language model running locally.

Classify this user feedback as one of:
- complaint
- question
- praise
- mixed

Also give one short reason.

Feedback:
"The dashboard is useful, but the loading time is too slow."
"""
    }
}