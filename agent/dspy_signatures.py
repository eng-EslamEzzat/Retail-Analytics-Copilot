"""DSPy signatures and modules for the retail analytics copilot."""
import dspy
from typing import Literal, Dict, Any, List, Optional, Tuple
import json
import re


class Router(dspy.Signature):
    """Route questions to appropriate handler: rag, sql, or hybrid."""
    question = dspy.InputField(desc="The user's question")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")


class Planner(dspy.Signature):
    """Extract constraints and requirements from question and retrieved docs."""
    question = dspy.InputField(desc="The user's question")
    retrieved_docs = dspy.InputField(desc="Retrieved document chunks")
    constraints = dspy.OutputField(desc="Extracted constraints: dates, KPIs, categories, entities as JSON")


class NLToSQL(dspy.Signature):
    """Generate SQLite query from natural language question and schema."""
    question = dspy.InputField(desc="The user's question")
    db_schema = dspy.InputField(desc="Database schema information")
    constraints = dspy.InputField(desc="Extracted constraints from planner")
    sql_query = dspy.OutputField(desc="Valid SQLite query only, no markdown, no explanation. Use double quotes for table names with spaces like \"Order Details\"")


class Synthesizer(dspy.Signature):
    """Synthesize final answer from SQL results and retrieved docs."""
    question = dspy.InputField(desc="The user's question")
    sql_results = dspy.InputField(desc="SQL query results (rows and columns)")
    retrieved_docs = dspy.InputField(desc="Retrieved document chunks")
    format_hint = dspy.InputField(desc="Expected output format")
    final_answer = dspy.OutputField(desc="Final answer matching format_hint exactly")
    citations = dspy.OutputField(desc="List of citations: DB tables and doc chunk IDs")


class RouterModule(dspy.Module):
    """Router module using DSPy."""
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(Router)
    
    def forward(self, question: str) -> str:
        result = self.router(question=question)
        route = result.route.lower().strip()
        # Normalize to one of the three options
        if "rag" in route or "document" in route:
            return "rag"
        elif "sql" in route or "database" in route:
            return "sql"
        else:
            return "hybrid"


class PlannerModule(dspy.Module):
    """Planner module to extract constraints."""
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(Planner)
    
    def forward(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        docs_str = "\n".join([
            f"[{doc.get('id', 'unknown')}]: {doc.get('content', '')}"
            for doc in retrieved_docs
        ])
        result = self.planner(question=question, retrieved_docs=docs_str)
        
        # Try to parse constraints as JSON
        try:
            constraints = json.loads(result.constraints)
        except:
            # Fallback: create simple dict
            constraints = {"raw": result.constraints}
        
        return constraints


class NLToSQLModule(dspy.Module):
    """Natural language to SQL module."""
    def __init__(self):
        super().__init__()
        self.sql_generator = dspy.ChainOfThought(NLToSQL)
    
    def forward(self, question: str, schema: str, constraints: Dict[str, Any]) -> str:
        constraints_str = json.dumps(constraints) if isinstance(constraints, dict) else str(constraints)
        result = self.sql_generator(
            question=question,
            db_schema=schema,
            constraints=constraints_str
        )
        
        # Handle case where model doesn't return sql_query field
        if not hasattr(result, 'sql_query') or result.sql_query is None or not result.sql_query.strip():
            # Try to extract SQL from reasoning or other fields
            if hasattr(result, 'reasoning'):
                reasoning = str(result.reasoning)
                # Try to find SQL between code blocks
                sql_match = re.search(r'```sql\s*(.*?)\s*```', reasoning, re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1).strip()
                else:
                    # Look for SELECT statements
                    select_match = re.search(
                        r'(SELECT\s+.*?;)', reasoning, re.DOTALL | re.IGNORECASE
                    )
                    if select_match:
                        sql = select_match.group(1).strip()
                    else:
                        # Fallback: return empty and let repair handle it
                        return ""
            else:
                return ""
        else:
            sql = result.sql_query.strip()
        
        # Clean SQL query (remove markdown code blocks if present)
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:].strip()
        sql = sql.strip()
        
        # Post-process: Fix unquoted table names with spaces
        sql = self._fix_table_names(sql)
        
        return sql
    
    def _fix_table_names(self, sql: str) -> str:
        """Fix unquoted table names that have spaces."""
        # Common Northwind table names that need quoting
        # Handle variations: OrderDetails, Order_Details, Order Details
        replacements = [
            # OrderDetails -> "Order Details"
            (r'\bOrderDetails\b', '"Order Details"'),
            (r'\borderdetails\b', '"Order Details"'),
            (r'\bORDERDETAILS\b', '"Order Details"'),
            # Order_Details -> "Order Details"
            (r'\bOrder_Details\b', '"Order Details"'),
            (r'\border_details\b', '"Order Details"'),
            # Order Details (unquoted) -> "Order Details"
            (r'(?<!")Order Details(?!")', '"Order Details"'),
            # Handle with dots/aliases
            (r'\bOrderDetails\.', '"Order Details".'),
            (r'\borderdetails\.', '"Order Details".'),
            (r'\.OrderDetails\b', '."Order Details"'),
            (r'\.orderdetails\b', '."Order Details"'),
        ]
        
        for pattern, replacement in replacements:
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        
        return sql


class SynthesizerModule(dspy.Module):
    """Synthesizer module to create final answer."""
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
        # Format SQL results
        sql_str = "No SQL results" if sql_results is None else json.dumps(sql_results, default=str)
        
        # Format retrieved docs
        docs_str = "\n".join([
            f"[{doc.get('id', 'unknown')}]: {doc.get('content', '')}"
            for doc in retrieved_docs
        ])
        
        result = self.synthesizer(
            question=question,
            sql_results=sql_str,
            retrieved_docs=docs_str,
            format_hint=format_hint
        )
        
        # Parse final answer based on format_hint
        answer = self._parse_answer(result.final_answer, format_hint)
        
        # Parse citations
        citations = self._parse_citations(result.citations)
        
        return answer, citations
    
    def _parse_answer(self, answer_str: str, format_hint: str) -> Any:
        """Parse answer string to match format_hint."""
        answer_str = answer_str.strip()
        
        # Try to extract JSON if present
        if "{" in answer_str or "[" in answer_str:
            try:
                # Extract JSON from markdown or text
                json_match = None
                if "```" in answer_str:
                    parts = answer_str.split("```")
                    for part in parts:
                        if "{" in part or "[" in part:
                            json_match = part.strip()
                            if json_match.startswith("json"):
                                json_match = json_match[4:].strip()
                            break
                else:
                    json_match = answer_str
                
                if json_match:
                    return json.loads(json_match)
            except:
                pass
        
        # Handle simple types
        if format_hint == "int":
            try:
                return int(float(answer_str.replace(",", "").split()[0]))
            except:
                return 0
        elif format_hint == "float":
            try:
                return round(float(answer_str.replace(",", "").split()[0]), 2)
            except:
                return 0.0
        
        return answer_str
    
    def _parse_citations(self, citations_str: str) -> List[str]:
        """Parse citations from string."""
        citations = []
        
        # Try JSON list
        try:
            citations = json.loads(citations_str)
            if isinstance(citations, list):
                return citations
        except:
            pass
        
        # Try to extract from text
        # Look for table names and chunk IDs
        import re
        # Table names (capitalized, common Northwind tables)
        tables = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]
        for table in tables:
            if table.lower() in citations_str.lower():
                citations.append(table)
        
        # Chunk IDs (pattern: filename::chunkN)
        chunk_pattern = r'(\w+)::chunk\d+'
        chunks = re.findall(chunk_pattern, citations_str)
        for chunk_match in re.finditer(chunk_pattern, citations_str):
            citations.append(chunk_match.group(0))
        
        return list(set(citations))  # Remove duplicates

