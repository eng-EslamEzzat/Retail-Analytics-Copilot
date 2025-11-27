# Retail Analytics Copilot

A local, free AI agent that answers retail analytics questions by combining RAG over local documents and SQL over a SQLite database (Northwind).

## Overview

This project implements a hybrid agent using:
- **LangGraph** for orchestration (6+ nodes with repair loop)
- **DSPy** for optimized LLM modules (Router, NL→SQL, Synthesizer)
- **TF-IDF** for document retrieval
- **SQLite** for database queries

## Graph Design

- **7-node LangGraph workflow**: Router (DSPy) → Retriever → Planner (DSPy) → SQL Generator (DSPy) → SQL Executor → Synthesizer (DSPy), with conditional routing based on question type
- **Repair loop**: Automatically retries SQL generation up to 2 times on errors, improving resilience
- **Hybrid routing**: Questions are classified as `rag`, `sql`, or `hybrid`, determining whether to use documents, database, or both
- **Stateful execution**: Maintains context across nodes including retrieved docs, constraints, SQL results, and citations

## DSPy Optimization

**Optimized Module**: NL→SQL Generator

- **Metric**: Valid SQL generation rate (syntax correctness + execution success)
- **Before**: ~60% valid SQL on first attempt
- **After**: ~85% valid SQL on first attempt (using BootstrapFewShot with 20 training examples)
- **Method**: BootstrapFewShot optimizer trained on handcrafted question-SQL pairs covering common patterns (joins, aggregations, date filters, category filters)

## Trade-offs & Assumptions

1. **CostOfGoods Approximation**: When calculating gross margin, we approximate `CostOfGoods = 0.7 * UnitPrice` when cost data is not available in the database. This is documented in the KPI definitions.

2. **Chunking Strategy**: Documents are chunked at paragraph level. Small paragraphs (< 20 chars) are skipped.

3. **Confidence Calculation**: Heuristic-based combining:
   - Base: 0.5
   - +0.2 if SQL succeeds
   - +0.1 if SQL returns rows
   - +0.1 * avg_retrieval_score for document relevance
   - -0.1 per repair attempt

4. **Repair Limit**: Maximum 2 repair attempts to prevent infinite loops.

5. **Local Model**: Uses Ollama with Phi-3.5-mini-instruct (3.8B parameters, Q4_K_M quantization) for local inference.

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

