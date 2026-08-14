import asyncio
import argparse
import time
from tests.utils import generate_test_video
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

console = Console()

async def submit_job(client: httpx.AsyncClient, video_path: str, fast_mode: bool = False):
    with open(video_path, "rb") as f:
        response = await client.post(
            "/jobs",
            files={"video": ("stress_test.mp4", f, "video/mp4")},
            data={"enable_3d": "false", "fast_mode": str(fast_mode).lower()}
        )
    if response.status_code != 200:
        return None
    return response.json().get("job_id")

async def poll_job(client: httpx.AsyncClient, job_id: str, progress, task_id):
    start_time = time.time()
    while True:
        response = await client.get(f"/jobs/{job_id}")
        if response.status_code != 200:
            progress.update(task_id, description=f"[red]Job {job_id} Error", completed=100)
            return "error", time.time() - start_time
            
        status = response.json().get("status")
        if status == "complete":
            progress.update(task_id, description=f"[green]Job {job_id} Complete", completed=100)
            return "complete", time.time() - start_time
        elif status == "failed":
            error_msg = response.json().get("error_message", "Unknown error")
            progress.update(task_id, description=f"[red]Job {job_id} Failed: {error_msg[:30]}", completed=100)
            return f"failed ({error_msg})", time.time() - start_time
            
        await asyncio.sleep(2)

async def run_load_test(base_url: str, num_jobs: int, complexity: str):
    console.print(f"[bold cyan]Generating {num_jobs} {complexity}-complexity video(s)...[/bold cyan]")
    
    # Generate one shared video for this run to save disk IO during generation
    video_path = generate_test_video(duration_seconds=5 if complexity == "low" else 15, complexity=complexity)
    
    console.print(f"[bold cyan]Submitting {num_jobs} concurrent jobs to {base_url}...[/bold cyan]")
    
    results = []
    
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        # Check health first
        try:
            health_res = await client.get("/health")
            if health_res.status_code != 200:
                console.print("[bold red]Server health check failed. Is the server running?[/bold red]")
                return
        except httpx.ConnectError:
            console.print(f"[bold red]Cannot connect to {base_url}. Ensure the server is running (e.g. ./run.sh)[/bold red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            job_tasks = []
            
            # Submit burst
            for i in range(num_jobs):
                job_id = await submit_job(client, str(video_path), fast_mode=(complexity=="low"))
                if job_id:
                    task_id = progress.add_task(f"[yellow]Job {job_id} Queued", total=100, completed=0)
                    job_tasks.append(poll_job(client, job_id, progress, task_id))
                else:
                    console.print(f"[red]Failed to submit job {i}[/red]")
            
            # Wait for all to complete
            outcomes = await asyncio.gather(*job_tasks)
            results = outcomes

    # Print Report
    console.print("\n[bold green]Load Test Complete. Summary:[/bold green]")
    table = Table(title="Job Execution Results")
    table.add_column("Status", justify="left", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Avg Time (s)", justify="right", style="green")

    status_counts = {}
    total_times = {}

    for status, duration in results:
        base_status = status.split(" ")[0]
        status_counts[base_status] = status_counts.get(base_status, 0) + 1
        total_times[base_status] = total_times.get(base_status, 0) + duration

    for status, count in status_counts.items():
        avg_time = total_times[status] / count
        table.add_row(status, str(count), f"{avg_time:.2f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniUI System Load Tester")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of the OmniUI API")
    parser.add_argument("--jobs", type=int, default=5, help="Number of concurrent jobs to submit")
    parser.add_argument("--complexity", type=str, choices=["low", "high"], default="low", help="Video complexity (low=fast, high=heavy load triggering all models)")
    
    args = parser.parse_args()
    
    asyncio.run(run_load_test(args.url, args.jobs, args.complexity))
