"""Upload benchmark results to HuggingFace dataset.

This module handles uploading summary JSONs to a HF dataset for leaderboard display.
Supports both individual and batch uploads with idempotency.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Set


from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from huggingface.config import HF_TOKEN, HF_RESULTS_DATASET
from huggingface.schema import summary_to_leaderboard_row, leaderboard_row_to_json


logger = logging.getLogger(__name__)


class ResultUploader:
    """Handles uploading benchmark results to HuggingFace dataset."""
    
    def __init__(self, token: Optional[str] = None, dataset_id: Optional[str] = None):
        """Initialize uploader.
        
        Args:
            token: HF token with write access. Falls back to HF_TOKEN env var.
            dataset_id: HF dataset ID. Falls back to HF_RESULTS_DATASET env var.
        """
        self.token = token or HF_TOKEN
        self.dataset_id = dataset_id or HF_RESULTS_DATASET
        
        if not self.token:
            raise ValueError(
                "HF_TOKEN must be provided or set in environment. "
                "Get a token at https://huggingface.co/settings/tokens"
            )
        
        if not self.dataset_id:
            raise ValueError(
                "HF_RESULTS_DATASET must be provided or set in environment."
            )
        
        self.api = HfApi(token=self.token)
        logger.info(f"Initialized uploader for dataset: {self.dataset_id}")
    
    def get_uploaded_results(self) -> Set[str]:
        """Get set of experiment IDs already uploaded to dataset.
        
        Returns:
            Set of experiment_id strings.
        """
        try:
            files = self.api.list_repo_files(
                repo_id=self.dataset_id,
                repo_type="dataset"
            )
            
            # Extract experiment IDs from filenames: results/{experiment_id}.json
            uploaded = set()
            for file_path in files:
                if file_path.startswith("results/") and file_path.endswith(".json"):
                    experiment_id = Path(file_path).stem
                    uploaded.add(experiment_id)
            
            logger.info(f"Found {len(uploaded)} existing results in dataset")
            return uploaded
            
        except RepositoryNotFoundError:
            logger.warning(f"Dataset {self.dataset_id} not found. Will create on first upload.")
            return set()
        except Exception as e:
            logger.error(f"Failed to list uploaded results: {e}")
            return set()
    
    def upload_result(
        self,
        summary_path: Path,
        force: bool = False,
        skip_if_exists: bool = True
    ) -> bool:
        """Upload a single result summary to HF dataset.
        
        Args:
            summary_path: Path to summary JSON file.
            force: Force upload even if already exists.
            skip_if_exists: Skip upload if already exists (idempotent).
            
        Returns:
            True if uploaded, False if skipped.
        """
        if not summary_path.exists():
            logger.error(f"Summary file not found: {summary_path}")
            return False
        
        # Load summary
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load summary {summary_path}: {e}")
            return False
        
        experiment_id = summary.get("experiment_id")
        if not experiment_id:
            logger.error(f"No experiment_id in summary: {summary_path}")
            return False
        
        # Check if already uploaded
        if skip_if_exists and not force:
            uploaded = self.get_uploaded_results()
            if experiment_id in uploaded:
                logger.info(f"Result {experiment_id} already uploaded, skipping")
                return False
        
        # Convert to leaderboard row
        try:
            row = summary_to_leaderboard_row(summary)
        except Exception as e:
            logger.error(f"Failed to convert summary to row: {e}")
            return False
        
        # Add original summary for reference
        row["_original_summary"] = summary
        
        # Serialize to JSON
        result_json = leaderboard_row_to_json(row)
        result_path = f"results/{experiment_id}.json"
        
        # Upload to dataset
        try:
            content = json.dumps(result_json, indent=2).encode("utf-8")
            
            self.api.upload_file(
                path_or_fileobj=content,
                path_in_repo=result_path,
                repo_id=self.dataset_id,
                repo_type="dataset",
                commit_message=f"Upload result: {experiment_id}"
            )
            
            logger.info(f"Successfully uploaded {experiment_id} to {result_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {experiment_id}: {e}")
            return False
    
    def upload_results_batch(
        self,
        summary_paths: List[Path],
        force: bool = False,
        skip_if_exists: bool = True
    ) -> dict:
        """Upload multiple results to HF dataset.
        
        Args:
            summary_paths: List of paths to summary JSON files.
            force: Force upload even if already exists.
            skip_if_exists: Skip upload if already exists (idempotent).
            
        Returns:
            Dictionary with counts: {uploaded: int, skipped: int, failed: int}
        """
        uploaded_count = 0
        skipped_count = 0
        failed_count = 0
        
        logger.info(f"Starting batch upload of {len(summary_paths)} results")
        
        for summary_path in summary_paths:
            result = self.upload_result(
                summary_path,
                force=force,
                skip_if_exists=skip_if_exists
            )
            
            if result:
                uploaded_count += 1
            elif summary_path.exists():
                # File exists but wasn't uploaded (likely skipped)
                skipped_count += 1
            else:
                failed_count += 1
        
        summary = {
            "uploaded": uploaded_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "total": len(summary_paths)
        }
        
        logger.info(
            f"Batch upload complete: {uploaded_count} uploaded, "
            f"{skipped_count} skipped, {failed_count} failed"
        )
        
        return summary


def upload_result(
    summary_path: Path,
    token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    force: bool = False
) -> bool:
    """Convenience function to upload a single result.
    
    Args:
        summary_path: Path to summary JSON file.
        token: HF token (optional, uses env var).
        dataset_id: HF dataset ID (optional, uses env var).
        force: Force upload even if already exists.
        
    Returns:
        True if uploaded successfully, False otherwise.
    """
    uploader = ResultUploader(token=token, dataset_id=dataset_id)
    return uploader.upload_result(summary_path, force=force)


def upload_results_batch(
    summary_paths: List[Path],
    token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    force: bool = False
) -> dict:
    """Convenience function to upload multiple results.
    
    Args:
        summary_paths: List of paths to summary JSON files.
        token: HF token (optional, uses env var).
        dataset_id: HF dataset ID (optional, uses env var).
        force: Force upload even if already exists.
        
    Returns:
        Dictionary with upload counts.
    """
    uploader = ResultUploader(token=token, dataset_id=dataset_id)
    return uploader.upload_results_batch(summary_paths, force=force)


def get_uploaded_results(
    token: Optional[str] = None,
    dataset_id: Optional[str] = None
) -> Set[str]:
    """Convenience function to get list of uploaded experiment IDs.
    
    Args:
        token: HF token (optional, uses env var).
        dataset_id: HF dataset ID (optional, uses env var).
        
    Returns:
        Set of experiment IDs already uploaded.
    """
    uploader = ResultUploader(token=token, dataset_id=dataset_id)
    return uploader.get_uploaded_results()
