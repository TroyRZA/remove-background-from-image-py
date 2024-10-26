import os
from PIL import Image
from rembg import remove
import argparse
import logging
from pathlib import Path
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from typing import List, Tuple
import time

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_image(input_path: str, output_path: str = None, device: str = None) -> Tuple[bool, float]:
    """
    Remove background from a single image with performance metrics.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image
        device: Device to use for processing ('cuda' or 'cpu')
    
    Returns:
        Tuple of (success_status, processing_time)
    """
    start_time = time.time()
    try:

        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_nobg{input_file.suffix}")
        

        input_image = Image.open(input_path)

        max_size = 2000 
        if max(input_image.size) > max_size:
            ratio = max_size / max(input_image.size)
            new_size = tuple(int(dim * ratio) for dim in input_image.size)
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)

        if device:
            with torch.device(device):
                output_image = remove(input_image)
        else:
            output_image = remove(input_image)

        output_image.save(output_path, format='PNG', optimize=True)
        
        processing_time = time.time() - start_time
        return True, processing_time
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        processing_time = time.time() - start_time
        return False, processing_time

def batch_process(file_paths: List[str], output_dir: Path, num_workers: int, use_gpu: bool) -> List[Tuple[str, bool, float]]:
    """
    Process multiple images in parallel using either ThreadPoolExecutor or ProcessPoolExecutor.
    
    Args:
        file_paths: List of input file paths
        output_dir: Output directory path
        num_workers: Number of worker processes/threads
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        List of tuples containing (file_path, success_status, processing_time)
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    output_paths = [output_dir / f"{Path(fp).stem}_nobg{Path(fp).suffix}" for fp in file_paths]

    Executor = ThreadPoolExecutor if device == 'cuda' else ProcessPoolExecutor
    
    results = []
    with Executor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, str(input_path), str(output_path), device)
            for input_path, output_path in zip(file_paths, output_paths)
        ]
        
        for input_path, future in zip(file_paths, futures):
            success, proc_time = future.result()
            results.append((input_path, success, proc_time))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Optimized background removal from images')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', help='Output path (optional)', default=None)
    parser.add_argument('--batch', action='store_true', help='Process entire directory')
    parser.add_argument('--workers', type=int, default=None, 
                      help='Number of worker processes (default: CPU count)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available')
    
    args = parser.parse_args()
    logger = setup_logging()

    if args.workers is None:
        args.workers = mp.cpu_count() if not args.gpu else 1
    
    logger.info(f"Using device: {'CUDA' if args.gpu and torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Number of workers: {args.workers}")
    
    start_time = time.time()
    
    if args.batch:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        file_paths = [f for f in input_path.iterdir() 
                     if f.suffix.lower() in image_extensions]
        
        results = batch_process(file_paths, output_path, args.workers, args.gpu)
        
        successful = sum(1 for _, success, _ in results if success)
        total_time = time.time() - start_time
        avg_time = sum(t for _, _, t in results) / len(results)
        
        logger.info(f"Processed {len(results)} images:")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(results) - successful}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per image: {avg_time:.2f}s")
        
    else:
        success, proc_time = process_image(args.input, args.output, 
                                         'cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Processing {'successful' if success else 'failed'}")
        logger.info(f"Processing time: {proc_time:.2f}s")

if __name__ == "__main__":
    main()