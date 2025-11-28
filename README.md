# Retail Analytics Copilot

A local, free AI agent that answers retail analytics questions by combining RAG over local documents and SQL over a SQLite database (Northwind).

## Overview

This project implements a hybrid agent using:
- **LangGraph** for orchestration (8 nodes with validation + repair loop)
- **DSPy** for optimized LLM modules (Router, Planner, NL→SQL, Synthesizer)
- **TF-IDF** for document retrieval
- **SQLite** for database queries

## Graph Design

- **8-node LangGraph workflow**: Router → Retriever → Planner → SQL Generator → SQL Executor → Synthesizer → Validator → End/Repair. The validator enforces type-safe outputs and citation coverage before finalizing.
- **Repair loop**: Automatically retries SQL generation up to 2 times when SQLite errors or validator failures occur.
- **Hybrid routing**: Manual + DSPy router forces doc-only questions (policy) to RAG, campaign/KPI questions to hybrid, and pure SQL asks to DB only.
- **Stateful execution**: Tracks retrieved chunks, campaign constraints, SQL traces, validator findings, and emits `traces/trace_<ts>.json` for auditing.
- **Typed synthesis**: Deterministic formatter converts SQL rows into the requested schema and injects both DB table names and `doc::chunkN` citations.

## DSPy Optimization

**Optimized Module**: NL→SQL Generator (BootstrapFewShot)

- **Metric**: Exact-match SQL on a 4-example handcrafted eval (joins, aggregations, date filters, category filters).
- **Before**: 0.25 exact-match (plain Chain-of-Thought).
- **After**: 1.00 exact-match post-optimization (training data embedded in `agent/dspy_signatures.py`).
- **Method**: BootstrapFewShot compiles the program once per run and logs `[DSPy] NL→SQL accuracy 0.25 → 1.00`.

## Trade-offs & Assumptions

1. **CostOfGoods Approximation**: When calculating gross margin, we approximate `CostOfGoods = 0.7 * UnitPrice` when cost data is not available in the database. This is documented in the KPI definitions.

2. **Chunking Strategy**: Documents are chunked at paragraph level. Small paragraphs (< 20 chars) are skipped.

3. **Confidence Calculation**: Base 0.5, +0.3 for successful SQL execution, +0.1 for returned rows, +0.1 when doc evidence is cited, −0.1 per repair attempt.
4. **Repair Limit**: Maximum 2 repair attempts (shared between SQL errors and validator-triggered fixes).
5. **Local Model**: Uses Ollama with Phi-3.5-mini-instruct (3.8B parameters, Q4_K_M quantization) for local inference.
6. **Date Shift Handling**: The Northwind database provided contains data from 2016-2018, whereas the original dataset (and the sample questions) refer to 1996-1998. The agent automatically maps "1997" to "2017" in SQL queries to ensure correct results.
7. **Robust JSON Parsing**: To handle the non-deterministic nature of smaller local models, the agent includes a robust parsing layer that can handle Python-style dictionaries (single quotes) and malformed JSON, falling back to `ast.literal_eval` when necessary.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama and pull model**:
   ```bash
   # Install from https://ollama.com
   ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
   ```

3. **Ensure database exists**:
   - Database should be at `data/northwind.sqlite`

4. **Run the agent**:
   ```bash
   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
   ```

## Project Structure

```
.
├── agent/
│   ├── graph_hybrid.py          # LangGraph implementation
│   ├── dspy_signatures.py        # DSPy modules (Router, Planner, NL→SQL, Synthesizer)
│   ├── rag/
│   │   └── retrieval.py         # TF-IDF retriever
│   └── tools/
│       └── sqlite_tool.py        # Database access & schema introspection
├── data/
│   └── northwind.sqlite         # Northwind sample database
├── docs/
│   ├── marketing_calendar.md
│   ├── kpi_definitions.md
│   ├── catalog.md
│   └── product_policy.md
├── sample_questions_hybrid_eval.jsonl
├── run_agent_hybrid.py           # Main entrypoint
├── traces/
│   └── trace_<timestamp>.json    # Execution logs/checkpoints
└── requirements.txt
```

## Output Format

Each question produces a JSON object:
```json
{
    "id": "question_id",
    "final_answer": <matches format_hint>,
    "sql": "<last executed SQL or empty>",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation",
    "citations": ["Orders", "Products", "kpi_definitions::chunk0"]
}
```

## Notes

- No external network calls at inference time
- All processing is local
- Compatible with CPU-only systems (16GB RAM recommended)
- Prompts are kept compact (< 1k tokens total)

