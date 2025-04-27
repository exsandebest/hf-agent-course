---
title: GAIA Benchmark Agent - Final Assessment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# AI Agent for GAIA Benchmark

**Final assessment for the Hugging Face AI Agents course**

This repository contains a fully implemented autonomous agent designed to solve the [GAIA benchmark](https://arxiv.org/abs/2403.08790) - level 1. The agent leverages large language models and a suite of external tools to tackle complex, real-world, multi-modal tasks. It is ready to run and submit answers to the GAIA evaluation server, and is deployable as a HuggingFace Space with a Gradio interface.

## 🏆 Project Summary
- **Purpose:** Automatically solve and submit answers for the GAIA benchmark, which evaluates generalist AI agents on tasks requiring reasoning, code execution, web search, data analysis, and more.
- **Features:**
  - Uses LLMs (OpenAI, HuggingFace, etc.) for reasoning and planning
  - Integrates multiple tools: web search, Wikipedia, Python code execution, YouTube transcript, and more
  - Handles file-based and multi-modal tasks
  - Submits results and displays scores in a user-friendly Gradio interface

## 🚀 How to Run

**On HuggingFace Spaces:**
- Log in with your HuggingFace account.
- Click "Run Evaluation & Submit All Answers" to evaluate the agent on the GAIA benchmark and see your score.

**Locally:**
```bash
pip install -r requirements.txt
python app.py
```

## 🧠 About GAIA
GAIA is a challenging benchmark for evaluating the capabilities of generalist AI agents on real-world, multi-step, and multi-modal tasks. Each task may require code execution, web search, data analysis, or other tool use. This agent is designed to autonomously solve such tasks and submit answers for evaluation.

## 🏗️ Architecture
- `app.py` — Gradio app and evaluation logic. Fetches questions, runs the agent, and submits answers
- `agent.py` — Main `Agent` class. Implements reasoning, tool use, and answer formatting
- `model.py` — Loads and manages LLM backends (OpenAI, HuggingFace, LiteLLM, etc.)
- `tools.py` — Implements external tools
- `utils/logger.py` — Logging utility
- `requirements.txt` — All dependencies for local and Spaces deployment

## 🔑 Environment Variables
Some models require API keys. Set these in your Space or local environment:
- `OPENAI_API_KEY` and `OPENAI_API_BASE` (for OpenAI models)
- `HUGGINGFACEHUB_API_TOKEN` (for HuggingFace Hub models)

## 📦 Dependencies
All required packages are listed in `requirements.txt`
