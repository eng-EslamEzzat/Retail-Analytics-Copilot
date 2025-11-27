"""LangGraph implementation for hybrid RAG + SQL agent."""
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
import json

from agent.dspy_signatures import (
    RouterModule, PlannerModule, NLToSQLModule, SynthesizerModule
)
from agent.rag.retrieval import TFIDFRetriever
from agent.tools.sqlite_tool import SQLiteTool


class AgentState(TypedDict):
    """State for the agent graph."""
    question: str
    format_hint: str
    route: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    sql_query: Optional[str]
    sql_results: Optional[List[Dict[str, Any]]]
    sql_error: Optional[str]
    sql_columns: List[str]
    final_answer: Any
    citations: List[str]
    explanation: str
    confidence: float
    repair_count: int
    trace: List[str]


class HybridAgent:
    """Hybrid RAG + SQL agent using LangGraph."""
    
    def __init__(self, db_path: str, docs_dir: str = "docs"):
        """Initialize agent with database and documents."""
        self.db_tool = SQLiteTool(db_path)
        self.retriever = TFIDFRetriever(docs_dir)
        
        # Initialize DSPy modules
        self.router = RouterModule()
        self.planner = PlannerModule()
        self.nl_to_sql = NLToSQLModule()
        self.synthesizer = SynthesizerModule()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_node)
        workflow.add_node("retriever", self._retrieve_node)
        workflow.add_node("planner", self._plan_node)
        workflow.add_node("sql_generator", self._sql_generate_node)
        workflow.add_node("sql_executor", self._sql_execute_node)
        workflow.add_node("synthesizer", self._synthesize_node)
        workflow.add_node("repair", self._repair_node)
        
        # Define edges
        workflow.set_entry_point("router")
        
        # Router decides next step
        workflow.add_conditional_edges(
            "router",
            self._route_condition,
            {
                "rag": "retriever",
                "sql": "sql_generator",
                "hybrid": "retriever"
            }
        )
        
        # After retrieval, go to planner (for hybrid) or synthesizer (for rag-only)
        workflow.add_conditional_edges(
            "retriever",
            self._after_retrieve_condition,
            {
                "plan": "planner",
                "synthesize": "synthesizer"
            }
        )
        
        # Planner goes to SQL generator
        workflow.add_edge("planner", "sql_generator")
        
        # SQL generator goes to executor
        workflow.add_edge("sql_generator", "sql_executor")
        
        # SQL executor checks if repair needed
        workflow.add_conditional_edges(
            "sql_executor",
            self._after_sql_condition,
            {
                "repair": "repair",
                "synthesize": "synthesizer"
            }
        )
        
        # Repair goes back to SQL generator
        workflow.add_edge("repair", "sql_generator")
        
        # Synthesizer ends
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _route_node(self, state: AgentState) -> AgentState:
        """Route question to appropriate handler."""
        state["trace"].append("Routing question...")
        route = self.router(question=state["question"])
        state["route"] = route
        state["trace"].append(f"Routed to: {route}")
        return state
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        state["trace"].append("Retrieving documents...")
        chunks = self.retriever.retrieve(state["question"], top_k=5)
        state["retrieved_docs"] = [chunk.to_dict() for chunk in chunks]
        state["trace"].append(f"Retrieved {len(chunks)} chunks")
        return state
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Extract constraints from question and docs."""
        state["trace"].append("Planning constraints...")
        constraints = self.planner(
            question=state["question"],
            retrieved_docs=state["retrieved_docs"]
        )
        state["constraints"] = constraints
        state["trace"].append(f"Extracted constraints: {json.dumps(constraints)}")
        return state
    
    def _sql_generate_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from question."""
        state["trace"].append("Generating SQL...")
        schema = self.db_tool.get_schema_string()
        constraints = state.get("constraints", {})
        
        sql = self.nl_to_sql(
            question=state["question"],
            schema=schema,
            constraints=constraints
        )
        state["sql_query"] = sql
        state["trace"].append(f"Generated SQL: {sql}")
        return state
    
    def _sql_execute_node(self, state: AgentState) -> AgentState:
        """Execute SQL query."""
        state["trace"].append("Executing SQL...")
        results, error, columns = self.db_tool.execute_query(state["sql_query"])
        
        if error:
            state["sql_error"] = error
            state["sql_results"] = None
            state["trace"].append(f"SQL Error: {error}")
        else:
            state["sql_results"] = results
            state["sql_columns"] = columns
            state["sql_error"] = None
            state["trace"].append(f"SQL returned {len(results)} rows")
        
        return state
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer."""
        state["trace"].append("Synthesizing answer...")
        
        answer, citations = self.synthesizer(
            question=state["question"],
            sql_results=state.get("sql_results"),
            retrieved_docs=state.get("retrieved_docs", []),
            format_hint=state["format_hint"]
        )
        
        state["final_answer"] = answer
        state["citations"] = citations
        
        # Calculate confidence
        state["confidence"] = self._calculate_confidence(state)
        
        # Generate explanation
        state["explanation"] = self._generate_explanation(state)
        
        state["trace"].append("Answer synthesized")
        return state
    
    def _repair_node(self, state: AgentState) -> AgentState:
        """Repair SQL query on error."""
        state["repair_count"] = state.get("repair_count", 0) + 1
        state["trace"].append(f"Repair attempt {state['repair_count']}")
        
        # Clear previous SQL error and try again
        # The planner might need to be re-run with error context
        if state["repair_count"] <= 2:
            # Update constraints with error information
            if "sql_error" in state and state["sql_error"]:
                error_info = {"previous_error": state["sql_error"], "previous_query": state["sql_query"]}
                if "constraints" in state:
                    state["constraints"].update(error_info)
                else:
                    state["constraints"] = error_info
        else:
            # Max repairs reached, synthesize with error
            state["trace"].append("Max repairs reached")
        
        return state
    
    def _route_condition(self, state: AgentState) -> str:
        """Condition after routing."""
        return state["route"] or "hybrid"
    
    def _after_retrieve_condition(self, state: AgentState) -> str:
        """Condition after retrieval."""
        if state["route"] == "rag":
            return "synthesize"
        else:
            return "plan"
    
    def _after_sql_condition(self, state: AgentState) -> str:
        """Condition after SQL execution."""
        repair_count = state.get("repair_count", 0)
        
        # Check if we need repair
        if state.get("sql_error") and repair_count < 2:
            return "repair"
        
        # Check if SQL results are invalid (empty when we expect data)
        if state.get("sql_results") is not None:
            if len(state["sql_results"]) == 0 and "top" in state["question"].lower():
                # Might be a query issue, but don't repair too much
                if repair_count < 1:
                    return "repair"
        
        return "synthesize"
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score."""
        confidence = 0.5  # Base confidence
        
        # Boost if SQL succeeded
        if state.get("sql_results") is not None and state.get("sql_error") is None:
            confidence += 0.2
            if len(state.get("sql_results", [])) > 0:
                confidence += 0.1
        
        # Boost if we have retrieved docs
        if state.get("retrieved_docs"):
            avg_score = sum(doc.get("score", 0) for doc in state["retrieved_docs"]) / len(state["retrieved_docs"])
            confidence += avg_score * 0.1
        
        # Reduce if we had to repair
        repair_count = state.get("repair_count", 0)
        confidence -= repair_count * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_explanation(self, state: AgentState) -> str:
        """Generate brief explanation."""
        parts = []
        
        if state.get("route") == "rag":
            parts.append("Answered using document retrieval.")
        elif state.get("route") == "sql":
            parts.append("Answered using database query.")
        else:
            parts.append("Answered using hybrid approach combining documents and database.")
        
        if state.get("sql_results"):
            parts.append(f"Query returned {len(state['sql_results'])} rows.")
        
        return " ".join(parts)
    
    def run(self, question: str, format_hint: str) -> Dict[str, Any]:
        """Run the agent on a question."""
        initial_state: AgentState = {
            "question": question,
            "format_hint": format_hint,
            "route": None,
            "retrieved_docs": [],
            "constraints": {},
            "sql_query": None,
            "sql_results": None,
            "sql_error": None,
            "sql_columns": [],
            "final_answer": None,
            "citations": [],
            "explanation": "",
            "confidence": 0.0,
            "repair_count": 0,
            "trace": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"]
        }

