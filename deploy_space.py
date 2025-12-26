"""Deploy HuggingFace Space for ScaleDown leaderboard.

This script uploads the hf_space/ directory to the HuggingFace Space repository.
It can be used to update the deployment as well.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from rich.console import Console


console = Console()


def deploy_space(
    space_id: str,
    token: str,
    space_dir: Path = Path("hf_space"),
    commit_message: str = "Update leaderboard Space",
) -> None:
    """Deploy Space to HuggingFace.
    
    Args:
        space_id: HuggingFace Space ID (e.g., "scaledown/compression-leaderboard").
        token: HuggingFace token with write access.
        space_dir: Local directory containing Space files.
        commit_message: Git commit message.
    """
    if not space_dir.exists():
        console.print(f"[red]✗[/red] Space directory not found: {space_dir}")
        return
    
    console.print(f"[bold]Deploying Space to {space_id}...[/bold]")
    
    try:
        api = HfApi(token=token)
        
        # Upload entire directory to Space
        api.upload_folder(
            folder_path=str(space_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message=commit_message,
        )
        
        console.print(f"[green]✓[/green] Successfully deployed Space to {space_id}")
        console.print(f"[cyan]→[/cyan] View at: https://huggingface.co/spaces/{space_id}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Deployment failed: {e}")
        raise


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Deploy ScaleDown leaderboard to HuggingFace Space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy using token from environment
  python deploy_space.py
  
  # Deploy with custom token
  python deploy_space.py --hf-token YOUR_TOKEN
  
  # Deploy with custom commit message
  python deploy_space.py --message "Add comparison view"
        """
    )
    
    parser.add_argument(
        "--space-id",
        type=str,
        default=os.getenv("HF_LEADERBOARD_SPACE", "scaledown/compression-leaderboard"),
        help="HuggingFace Space ID (default: from HF_LEADERBOARD_SPACE env var)"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token (default: from HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--space-dir",
        type=Path,
        default=Path("hf_space"),
        help="Local Space directory (default: hf_space/)"
    )
    
    parser.add_argument(
        "--message",
        type=str,
        default="Update leaderboard Space",
        help="Git commit message"
    )
    
    args = parser.parse_args()
    
    if not args.hf_token:
        console.print("[red]✗[/red] HF_TOKEN is required. Set via --hf-token or HF_TOKEN environment variable.")
        return
    
    deploy_space(
        space_id=args.space_id,
        token=args.hf_token,
        space_dir=args.space_dir,
        commit_message=args.message,
    )


if __name__ == "__main__":
    main()
