# tests/run_full_ingestion.py

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from ingestion.ingestion_pipeline import IngestionPipeline
from models.document_element import DocumentElement

# Load environment variables
load_dotenv()

# Configure logging (clean, no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingestion_run.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


def save_elements_for_chunking(elements: List[DocumentElement], output_dir: Path, filename: str):
    """Save processed elements in a clean format ready for chunking."""
    doc_dir = output_dir / filename
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Save elements as JSON
    elements_path = doc_dir / "elements.json"
    with open(elements_path, "w", encoding="utf-8") as f:
        json.dump([el.model_dump() for el in elements], f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "total_elements": len(elements),
        "content_type_distribution": {},
        "figures_with_vision": 0,
        "tables_with_structure": 0,
        "average_quality_score": 0.0
    }

    quality_scores = []
    for el in elements:
        ctype = el.content_type
        summary["content_type_distribution"][ctype] = summary["content_type_distribution"].get(ctype, 0) + 1
        
        if el.content_type == "figure" and el.vision_summary:
            summary["figures_with_vision"] += 1
        
        if el.content_type == "table" and el.structured_content:
            summary["tables_with_structure"] += 1
        
        quality_scores.append(el.quality_score)

    if quality_scores:
        summary["average_quality_score"] = round(sum(quality_scores) / len(quality_scores), 3)

    summary_path = doc_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved processed data for {filename} -> {doc_dir}")


def run_full_ingestion(input_dir: str, output_dir: str = "processed_data"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"Starting full ingestion on: {input_dir}")
    logger.info(f"Output will be saved to: {output_path}")

    # Initialize pipeline
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        logger.error("LLAMA_CLOUD_API_KEY not found in .env file")
        return

    pipeline = IngestionPipeline(
        llama_parse_api_key=api_key,
        vision_model="llama3.2-vision:latest",
        enable_cache=True,
        parser_version="v2.1"
    )

    # Get all supported files
    supported_files = []
    for ext in pipeline.SUPPORTED_FORMATS:
        supported_files.extend(input_path.glob(f"*{ext}"))

    if not supported_files:
        logger.warning("No supported files found in the directory")
        return

    logger.info(f"Found {len(supported_files)} files to process")

    total_elements = 0
    successful_files = 0
    failed_files = []

    start_time = time.time()

    for file_path in supported_files:
        filename = file_path.stem
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {file_path.name}")

        try:
            elements: List[DocumentElement] = pipeline.process_file(str(file_path))

            if elements:
                save_elements_for_chunking(elements, output_path, filename)
                total_elements += len(elements)
                successful_files += 1

                # Print summary for this file
                content_dist = {}
                for el in elements:
                    content_dist[el.content_type] = content_dist.get(el.content_type, 0) + 1

                logger.info(f"  -> Elements: {len(elements)}")
                logger.info(f"  -> Distribution: {content_dist}")

            else:
                logger.warning(f"  -> No elements extracted from {file_path.name}")

        except Exception as e:
            logger.error(f"  -> FAILED to process {file_path.name}: {str(e)}")
            failed_files.append(file_path.name)

    # Final Report
    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("INGESTION RUN COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total files processed     : {len(supported_files)}")
    logger.info(f"Successful files          : {successful_files}")
    logger.info(f"Failed files              : {len(failed_files)}")
    if failed_files:
        logger.info(f"Failed file names         : {failed_files}")
    logger.info(f"Total elements extracted  : {total_elements}")
    logger.info(f"Total time taken          : {total_time:.2f} seconds")
    logger.info(f"Average time per file     : {total_time / len(supported_files):.2f} seconds")
    logger.info(f"Processed data location   : {output_path}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    # Change this path if needed
    input_directory = "test_data/research_papers"
    run_full_ingestion(input_directory)