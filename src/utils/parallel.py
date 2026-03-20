"""Parallel processing utilities for benchmark execution.

This module provides thread-safe parallel execution of benchmark examples
using ThreadPoolExecutor, with rate limit handling and integrity guarantees.
"""
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Callable, Set, Optional

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Thread-safe parallel processor for benchmark examples.
    
    Handles concurrent execution with:
    - Thread-safe result collection and file writing
    - Rate limit detection and backoff
    - Example-level error isolation
    - Progress tracking
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        retry_attempts: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
    ):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum concurrent workers (default: 5)
            retry_attempts: Number of retry attempts on rate limit (default: 5)
            retry_base_delay: Base delay for exponential backoff in seconds (default: 2.0)
            retry_max_delay: Maximum delay cap in seconds (default: 60.0)
        """
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        
        # Thread-safe locks for shared resources
        self._file_lock = threading.Lock()
        self._processed_ids_lock = threading.Lock()
        self._global_backoff_lock = threading.Lock()
        
        # Global backoff state (shared across all threads)
        self._global_backoff_until = 0.0
        
        logger.info(
            f"Initialized ParallelProcessor: max_workers={max_workers}, "
            f"retry_attempts={retry_attempts}"
        )
    
    def process_examples(
        self,
        examples: List[Dict[str, Any]],
        process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        output_file: Path,
        existing_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        """Process examples in parallel with thread-safe writes.
        
        Args:
            examples: List of example dictionaries to process
            process_fn: Function that processes a single example and returns result dict
            output_file: Path to output JSONL file for thread-safe writes
            existing_ids: Set of already-processed IDs (thread-safe access managed internally)
        
        Returns:
            List of result dictionaries (in completion order)
        """
        results = []
        completed = 0
        total = len(examples)
        
        logger.info(f"Starting parallel processing of {total} examples with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_example = {}
            for example in examples:
                example_id = example["id"]
                
                # Check if already processed (thread-safe)
                with self._processed_ids_lock:
                    if example_id in existing_ids:
                        logger.debug(f"Skipping already processed example: {example_id}")
                        completed += 1
                        continue
                
                # Submit task
                future = executor.submit(
                    self._process_with_retry,
                    example,
                    process_fn,
                )
                future_to_example[future] = example
            
            # Collect results as they complete
            for future in as_completed(future_to_example):
                example = future_to_example[future]
                example_id = example["id"]
                
                try:
                    result = future.result()
                    
                    if result is not None:
                        # Thread-safe: write to file and update tracking
                        with self._file_lock:
                            with open(output_file, "a") as f:
                                f.write(json.dumps(result) + "\n")
                                f.flush()
                        
                        with self._processed_ids_lock:
                            existing_ids.add(example_id)
                        
                        results.append(result)
                        completed += 1
                        
                        if completed % 10 == 0 or completed == total:
                            logger.info(f"Progress: {completed}/{total} examples completed")
                    else:
                        logger.error(f"Example {example_id} failed after all retries")
                        completed += 1
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing example {example_id}: {e}", exc_info=True)
                    completed += 1
        
        logger.info(f"Parallel processing completed: {len(results)}/{total} successful")
        return results
    
    def _process_with_retry(
        self,
        example: Dict[str, Any],
        process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Process single example with retry logic.
        
        Args:
            example: Example dictionary to process
            process_fn: Processing function
        
        Returns:
            Result dictionary or None if all retries failed
        """
        example_id = example["id"]
        
        for attempt in range(self.retry_attempts):
            try:
                # Check if we should wait due to global backoff
                self._wait_for_global_backoff()
                
                # Process the example
                result = process_fn(example)
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str or
                    "429" in error_str or
                    "too many requests" in error_str or
                    "quota exceeded" in error_str
                )
                
                if is_rate_limit:
                    # Rate limit error: apply backoff
                    if attempt < self.retry_attempts - 1:
                        delay = min(
                            self.retry_base_delay * (2 ** attempt),
                            self.retry_max_delay
                        )
                        logger.warning(
                            f"Rate limit hit for example {example_id}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_attempts})"
                        )
                        
                        # Set global backoff to prevent other threads from hammering API
                        with self._global_backoff_lock:
                            backoff_until = time.time() + delay
                            if backoff_until > self._global_backoff_until:
                                self._global_backoff_until = backoff_until
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Rate limit exceeded for example {example_id} "
                            f"after {self.retry_attempts} attempts"
                        )
                        return None
                else:
                    # Non-rate-limit error: log and retry with shorter delay
                    if attempt < self.retry_attempts - 1:
                        delay = self.retry_base_delay
                        logger.warning(
                            f"Error processing example {example_id}: {e}. "
                            f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_attempts})"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to process example {example_id} after {self.retry_attempts} attempts: {e}"
                        )
                        return None
        
        return None
    
    def _wait_for_global_backoff(self):
        """Wait if global backoff is active."""
        with self._global_backoff_lock:
            wait_time = self._global_backoff_until - time.time()
            if wait_time > 0:
                logger.debug(f"Global backoff active, waiting {wait_time:.1f}s")
                time.sleep(wait_time)


def process_examples_parallel(
    examples: List[Dict[str, Any]],
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    output_file: Path,
    existing_ids: Set[str],
    max_workers: int = 5,
    retry_attempts: int = 5,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 60.0,
) -> List[Dict[str, Any]]:
    """Convenience function for parallel processing of examples.
    
    Args:
        examples: List of example dictionaries to process
        process_fn: Function that processes a single example and returns result dict
        output_file: Path to output JSONL file for thread-safe writes
        existing_ids: Set of already-processed IDs
        max_workers: Maximum concurrent workers (default: 5)
        retry_attempts: Number of retry attempts on rate limit (default: 5)
        retry_base_delay: Base delay for exponential backoff in seconds (default: 2.0)
        retry_max_delay: Maximum delay cap in seconds (default: 60.0)
    
    Returns:
        List of result dictionaries (in completion order)
    """
    processor = ParallelProcessor(
        max_workers=max_workers,
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    
    return processor.process_examples(
        examples=examples,
        process_fn=process_fn,
        output_file=output_file,
        existing_ids=existing_ids,
    )
