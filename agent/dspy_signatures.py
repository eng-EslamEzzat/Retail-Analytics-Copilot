"""DSPy signatures and modules for the retail analytics copilot."""
import json
import re
import ast
from typing import Dict, Any, List, Optional, Tuple

import dspy


SCHEMA_SNIPPET = """
Table: Orders (OrderID, CustomerID, EmployeeID, OrderDate, RequiredDate, ShippedDate, ShipCountry)
Table: "Order Details" (OrderID, ProductID, UnitPrice, Quantity, Discount)
Table: Products (ProductID, ProductName, SupplierID, CategoryID, QuantityPerUnit, UnitPrice)
Table: Categories (CategoryID, CategoryName)
Table: Customers (CustomerID, CompanyName, Country)
"""


class Router(dspy.Signature):
    """Route questions to appropriate handler: rag, sql, or hybrid."""
    question = dspy.InputField(desc="The user's question")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")


class Planner(dspy.Signature):
    """Extract constraints and requirements from question and retrieved docs."""
    question = dspy.InputField(desc="The user's question")
    retrieved_docs = dspy.InputField(desc="Retrieved document chunks")
    constraints = dspy.OutputField(desc="Extracted constraints: dates, KPIs, categories, entities as JSON. Ensure valid JSON with double quotes.")


class NLToSQL(dspy.Signature):
    """Generate SQLite query from natural language question and schema."""
    question = dspy.InputField(desc="The user's question")
    db_schema = dspy.InputField(desc="Database schema information")
    constraints = dspy.InputField(desc="Extracted constraints from planner")
    sql_query = dspy.OutputField(desc='Return valid SQLite only. Use double quotes for tables like "Order Details".')


class Synthesizer(dspy.Signature):
    """Synthesize final answer from SQL results and retrieved docs."""
    question = dspy.InputField(desc="The user's question")
    sql_results = dspy.InputField(desc="SQL query results (rows and columns)")
    retrieved_docs = dspy.InputField(desc="Retrieved document chunks")
    format_hint = dspy.InputField(desc="Expected output format")
    final_answer = dspy.OutputField(desc="Final answer matching format_hint exactly. If returning JSON, ensure it is valid JSON with double quotes.")
    citations = dspy.OutputField(desc="List of citations: DB tables and doc chunk IDs")


class RouterModule(dspy.Module):
    """Router module using DSPy."""

    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(Router)

    def forward(self, question: str) -> str:
        result = self.router(question=question)
        route = result.route.lower().strip()
        if "rag" in route or "document" in route:
            return "rag"
        if "sql" in route or "database" in route:
            return "sql"
        return "hybrid"


class PlannerModule(dspy.Module):
    """Planner module to extract constraints."""

    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(Planner)

    def forward(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        docs_str = "\n".join(
            f"[{doc.get('id', 'unknown')}]: {doc.get('content', '')}"
            for doc in retrieved_docs
        )
        result = self.planner(question=question, retrieved_docs=docs_str)
        try:
            constraints = json.loads(result.constraints)
        except Exception:
            try:
                constraints = ast.literal_eval(result.constraints)
            except Exception:
                constraints = {"raw": result.constraints}
        return constraints


def _normalize_sql(sql: str) -> str:
    sql = sql or ""
    sql = sql.replace("\n", " ")
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip().lower()


def _sql_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    gold = _normalize_sql(example.sql_query)
    pred = _normalize_sql(getattr(prediction, "sql_query", ""))
    return 1.0 if gold == pred and gold else 0.0


class NLToSQLModule(dspy.Module):
    """Natural language to SQL module with BootstrapFewShot optimization."""

    def __init__(self):
        super().__init__()
        self.base_program = dspy.ChainOfThought(NLToSQL)
        self.trainset = self._build_trainset()
        self.optimized_program = self.base_program
        self.metrics: Dict[str, float] = {"baseline": 0.0, "optimized": 0.0}
        if self.trainset:
            self._optimize()

    def forward(self, question: str, schema: str, constraints: Dict[str, Any]) -> str:
        constraints_str = json.dumps(constraints) if isinstance(constraints, dict) else str(constraints)
        program = self.optimized_program or self.base_program
        result = program(
            question=question,
            db_schema=schema,
            constraints=constraints_str
        )
        sql = getattr(result, "sql_query", "") or ""
        sql = self._extract_sql(sql, getattr(result, "reasoning", ""))
        sql = self._fix_table_names(sql)
        return sql.strip()

    def _build_trainset(self) -> List[dspy.Example]:
        examples = [
            dspy.Example(
                question="Top 3 products by revenue all time",
                db_schema=SCHEMA_SNIPPET,
                constraints=json.dumps({"metric": "revenue"}),
                sql_query=(
                    'SELECT p.ProductName AS product, '
                    'SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue '
                    'FROM "Order Details" od '
                    'JOIN Products p ON od.ProductID = p.ProductID '
                    'GROUP BY p.ProductID '
                    'ORDER BY revenue DESC '
                    'LIMIT 3'
                )
            ).with_inputs("question", "db_schema", "constraints"),
            dspy.Example(
                question="Average order value for Winter Classics 1997",
                db_schema=SCHEMA_SNIPPET,
                constraints=json.dumps({"date_range": {"start": "1997-12-01", "end": "1997-12-31"}}),
                sql_query=(
                    'SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) '
                    '/ COUNT(DISTINCT o.OrderID), 2) AS value '
                    'FROM Orders o '
                    'JOIN "Order Details" od ON o.OrderID = od.OrderID '
                    'WHERE o.OrderDate BETWEEN "1997-12-01" AND "1997-12-31"'
                )
            ).with_inputs("question", "db_schema", "constraints"),
            dspy.Example(
                question="Total beverage revenue in June 1997",
                db_schema=SCHEMA_SNIPPET,
                constraints=json.dumps({"category": "Beverages", "date_range": {"start": "1997-06-01", "end": "1997-06-30"}}),
                sql_query=(
                    'SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS revenue '
                    'FROM Orders o '
                    'JOIN "Order Details" od ON o.OrderID = od.OrderID '
                    'JOIN Products p ON od.ProductID = p.ProductID '
                    'JOIN Categories c ON p.CategoryID = c.CategoryID '
                    'WHERE c.CategoryName = "Beverages" '
                    'AND o.OrderDate BETWEEN "1997-06-01" AND "1997-06-30"'
                )
            ).with_inputs("question", "db_schema", "constraints"),
            dspy.Example(
                question="Top customer by gross margin 1997",
                db_schema=SCHEMA_SNIPPET,
                constraints=json.dumps({"metric": "margin", "year": "1997"}),
                sql_query=(
                    'SELECT c.CompanyName AS customer, '
                    'SUM((od.UnitPrice - (od.UnitPrice * 0.7)) * od.Quantity * (1 - od.Discount)) AS margin '
                    'FROM Orders o '
                    'JOIN "Order Details" od ON o.OrderID = od.OrderID '
                    'JOIN Customers c ON o.CustomerID = c.CustomerID '
                    'WHERE strftime("%Y", o.OrderDate) = "1997" '
                    'GROUP BY c.CustomerID '
                    'ORDER BY margin DESC '
                    'LIMIT 1'
                )
            ).with_inputs("question", "db_schema", "constraints"),
        ]
        return examples

    def _optimize(self):
        baseline = self._evaluate_program(self.base_program)
        optimizer = dspy.BootstrapFewShot(
            metric=_sql_metric,
            max_bootstrapped_demos=6,
            max_labeled_demos=len(self.trainset),
        )
        self.optimized_program = optimizer.compile(
            student=self.base_program,
            trainset=self.trainset
        )
        optimized = self._evaluate_program(self.optimized_program)
        self.metrics = {"baseline": baseline, "optimized": optimized}
        print(f"[DSPy] NL→SQL accuracy {baseline:.2f} → {optimized:.2f}")

    def _evaluate_program(self, program) -> float:
        if not self.trainset:
            return 0.0
        hits = 0
        for example in self.trainset:
            pred = program(
                question=example.question,
                db_schema=example.db_schema,
                constraints=example.constraints
            )
            sql = getattr(pred, "sql_query", "")
            if _normalize_sql(sql) == _normalize_sql(example.sql_query):
                hits += 1
        return hits / len(self.trainset)

    def _extract_sql(self, sql: str, reasoning: str) -> str:
        sql = sql or ""
        if sql.strip():
            return sql
        reasoning = reasoning or ""
        fenced = re.search(r'```sql\s*(.*?)```', reasoning, re.IGNORECASE | re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        select_stmt = re.search(r'(select\s.+)', reasoning, re.IGNORECASE | re.DOTALL)
        if select_stmt:
            return select_stmt.group(1).strip()
        return ""

    def _fix_table_names(self, sql: str) -> str:
        replacements = [
            (r'\bOrderDetails\b', '"Order Details"'),
            (r'\borderdetails\b', '"Order Details"'),
            (r'\bOrder_Details\b', '"Order Details"'),
            (r'(?<!")Order Details(?!")', '"Order Details"'),
            (r'\.OrderDetails\b', '."Order Details"'),
            (r'\.orderdetails\b', '."Order Details"'),
        ]
        for pattern, replacement in replacements:
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        return sql


class SynthesizerModule(dspy.Module):
    """Synthesizer module retained for optional LLM-based reasoning."""

    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(Synthesizer)

    def forward(
        self,
        question: str,
        sql_results: Optional[List[Dict[str, Any]]],
        retrieved_docs: List[Dict[str, Any]],
        format_hint: str
    ) -> Tuple[Any, List[str]]:
        sql_str = "No SQL results" if sql_results is None else json.dumps(sql_results, default=str)
        docs_str = "\n".join(
            f"[{doc.get('id', 'unknown')}]: {doc.get('content', '')}"
            for doc in retrieved_docs
        )
        result = self.synthesizer(
            question=question,
            sql_results=sql_str,
            retrieved_docs=docs_str,
            format_hint=format_hint
        )
        answer = self._parse_answer(result.final_answer, format_hint)
        citations = self._parse_citations(result.citations)
        return answer, citations

    def _parse_answer(self, answer_str: str, format_hint: str) -> Any:
        answer_str = (answer_str or "").strip()
        if "{" in answer_str or "[" in answer_str:
            try:
                candidate = answer_str
                if "```" in answer_str:
                    chunks = [chunk.strip() for chunk in answer_str.split("```") if chunk.strip()]
                    for chunk in chunks:
                        if chunk.startswith("json"):
                            chunk = chunk[4:].strip()
                        if chunk.startswith("{") or chunk.startswith("["):
                            candidate = chunk
                            break
                parsed = json.loads(candidate)
                return parsed
            except Exception:
                # Fallback for Python-style dicts with single quotes
                try:
                    return ast.literal_eval(candidate)
                except Exception:
                    pass
        if format_hint == "int":
            try:
                return int(float(answer_str.split()[0]))
            except Exception:
                return 0
        if format_hint == "float":
            try:
                return round(float(answer_str.split()[0]), 2)
            except Exception:
                return 0.0
        return answer_str

    def _parse_citations(self, citations_str: str) -> List[str]:
        try:
            parsed = json.loads(citations_str)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        tables = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]
        found: List[str] = []
        for table in tables:
            if table.lower() in citations_str.lower():
                found.append(table)
        for chunk in re.findall(r'\w+::chunk\d+', citations_str):
            found.append(chunk)
        return list(dict.fromkeys(found))

