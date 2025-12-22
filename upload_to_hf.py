#!/usr/bin/env python
"""Standalone CLI tool for uploading experiment results to HuggingFace."""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from huggingface.uploader import ResultUploader


console = Console()


def find_result_files(
    results_dir: Path,
    pattern: Optional[str] = None,
) -> List[Path]:
    """Find summary result files.
    
    Args:
        results_dir: Directory containing results.
        pattern: Optional glob pattern for filtering files.
        
    Returns:
        List of summary JSON file paths.
    """
    summaries_dir = results_dir / "summaries"
    if not summaries_dir.exists():
        console.print(f"[red]✗[/red] Summaries directory not found: {summaries_dir}")
        return []
    
    # Find JSON summary files
    if pattern:
        files = list(summaries_dir.glob(pattern))
    else:
        files = list(summaries_dir.glob("*.json"))
    
    return sorted(files)


def upload_results(
    files: List[Path],
    dry_run: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """Upload result files to HuggingFace.
    
    Args:
        files: List of summary JSON files to upload.
        dry_run: If True, validate but don't upload.
        hf_token: Optional HuggingFace token override.
    """
    if not files:
        console.print("[yellow]No files to upload[/yellow]")
        return
    
    # Initialize uploader
    try:
        uploader = ResultUploader(token=hf_token)
    except ValueError as e:
        console.print(f"[red]✗[/red] Failed to initialize uploader: {e}")
        sys.exit(1)
    
    # Display files to upload
    table = Table(title="Files to Upload")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    
    for file_path in files:
        table.add_row(str(file_path.name), "Ready")
    
    console.print(table)
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - no files will be uploaded[/yellow]")
        return
    
    # Upload files
    console.print(f"\n[bold]Uploading {len(files)} files...[/bold]")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            was_uploaded = uploader.upload_result(file_path)
            if was_uploaded:
                console.print(f"[green]✓[/green] Uploaded: {file_path.name}")
                success_count += 1
            else:
                console.print(f"[yellow]↷[/yellow] Skipped (already exists): {file_path.name}")
                skip_count += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Error uploading {file_path.name}: {e}")
            error_count += 1
    
    # Summary
    console.print("\n[bold]Upload Summary:[/bold]")
    console.print(f"  [green]✓[/green] Uploaded: {success_count}")
    console.print(f"  [yellow]↷[/yellow] Skipped: {skip_count}")
    console.print(f"  [red]✗[/red] Errors: {error_count}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Upload experiment results to HuggingFace leaderboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all results from default directory
  python upload_to_hf.py --all
  
  # Upload specific file
  python upload_to_hf.py --file results/summaries/hotpotqa_baseline_openai_*.json
  
  # Dry run to preview uploads
  python upload_to_hf.py --all --dry-run
  
  # Upload with custom token
  python upload_to_hf.py --all --hf-token YOUR_TOKEN
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing results (default: results/)"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Upload specific file (supports glob patterns, e.g., '*baseline*.json')"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all summary JSON files from results/summaries/"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview uploads without actually uploading"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token (overrides HF_TOKEN environment variable)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.file and not args.all:
        parser.error("Must specify either --file or --all")
    
    if args.file and args.all:
        parser.error("Cannot specify both --file and --all")
    
    # Find files to upload
    if args.all:
        files = find_result_files(args.results_dir)
    else:
        files = find_result_files(args.results_dir, pattern=args.file)
    
    if not files:
        console.print("[red]✗[/red] No matching files found")
        sys.exit(1)
    
    # Upload results
    upload_results(
        files=files,
        dry_run=args.dry_run,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
