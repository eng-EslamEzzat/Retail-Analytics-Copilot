"""Main entrypoint for the retail analytics copilot."""
import json
import click
from pathlib import Path
import dspy
from rich.console import Console
from rich.progress import Progress

from agent.graph_hybrid import HybridAgent

console = Console()


def setup_dspy():
    """Setup DSPy with local Ollama model."""
    import requests
    
    # First check if Ollama service is running
    try:
        response = requests.get(
            "http://localhost:11434/api/tags", timeout=2
        )
        if response.status_code != 200:
            console.print("[red]✗[/red] Ollama is not responding correctly")
            return False
    except requests.exceptions.RequestException:
        console.print("[red]✗[/red] Cannot connect to Ollama service")
        console.print("[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Start Ollama: ollama serve")
        console.print("  2. Or ensure Ollama service is running")
        return False
    
    # Try to use Ollama with DSPy
    try:
        # Use the model name as specified in the assignment
        model_name = "ollama/phi3.5:3.8b-mini-instruct-q4_K_M"
        lm = dspy.LM(model=model_name, api_base="http://localhost:11434")
        dspy.configure(lm=lm)
        console.print("[green]✓[/green] Connected to Ollama")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to setup DSPy: {e}")
        console.print("[yellow]Troubleshooting:[/yellow]")
        cmd = "ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
        console.print(f"  1. Pull the model: {cmd}")
        console.print("  2. Or try: ollama pull phi3.5")
        console.print("  3. Verify with: ollama list")
        return False


@click.command()
@click.option(
    "--batch", required=True, help="Path to JSONL file with questions"
)
@click.option("--out", required=True, help="Path to output JSONL file")
def main(batch: str, out: str):
    """Run the retail analytics copilot on a batch of questions."""
    console.print("[bold blue]Retail Analytics Copilot[/bold blue]")
    console.print("=" * 50)
    
    # Setup DSPy
    if not setup_dspy():
        console.print("[red]Exiting: DSPy setup failed[/red]")
        return
    
    # Check files
    batch_path = Path(batch)
    if not batch_path.exists():
        console.print(f"[red]Error:[/red] Batch file not found: {batch}")
        return
    
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    db_path = "data/northwind.sqlite"
    if not Path(db_path).exists():
        console.print(f"[red]Error:[/red] Database not found: {db_path}")
        return
    
    console.print("[green]✓[/green] Initializing agent...")
    try:
        agent = HybridAgent(db_path=db_path, docs_dir="docs")
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to initialize agent: {e}")
        import traceback
        console.print(traceback.format_exc())
        return

    # Load questions
    questions = []
    try:
        with open(batch_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        err_msg = (
                            f"[red]Error:[/red] Invalid JSON on line "
                            f"{line_num}: {e}"
                        )
                        console.print(err_msg)
                        if len(line) > 100:
                            line_preview = line[:100] + "..."
                        else:
                            line_preview = line
                        console.print(f"  Line content: {line_preview}")
                        return
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read batch file: {e}")
        return

    console.print(f"[green]✓[/green] Loaded {len(questions)} questions")

    # Process questions
    results = []
    with Progress() as progress:
        task_desc = "[cyan]Processing questions...[/cyan]"
        task = progress.add_task(task_desc, total=len(questions))
        
        for q in questions:
            q_id = q.get("id", "unknown")
            question = q.get("question", "")
            format_hint = q.get("format_hint", "")
            
            console.print(f"\n[bold]Question {q_id}:[/bold] {question}")
            
            try:
                result = agent.run(question=question, format_hint=format_hint)
                
                output = {
                    "id": q_id,
                    "final_answer": result["final_answer"],
                    "sql": result["sql"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"],
                    "citations": result["citations"]
                }
                
                results.append(output)
                ans = result['final_answer']
                console.print(f"[green]✓[/green] Answer: {ans}")
                conf = result['confidence']
                console.print(f"   Confidence: {conf:.2f}")
                cites = ', '.join(result['citations'])
                console.print(f"   Citations: {cites}")
                
            except Exception as e:
                console.print(f"[red]✗[/red] Error processing question: {e}")
                output = {
                    "id": q_id,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                }
                results.append(output)
            
            progress.update(task, advance=1)
    
    # Write results
    with open(out_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    console.print(f"\n[green]✓[/green] Results written to: {out_path}")
    console.print(f"[green]✓[/green] Processed {len(results)} questions")


if __name__ == "__main__":
    main()

