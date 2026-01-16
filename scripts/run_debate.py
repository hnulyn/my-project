#!/usr/bin/env python3
"""
Multi-process debate running script - supports parallel processing of multiple claims
"""

import argparse
import sys
import os
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.debate import Debate
from src.models.claim import Claim, normalize_label
from src.utils.config_manager import ConfigManager
from src.utils.openai_client import OpenAIClient
from src.utils.file_utils import FileUtils


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run fact-checking debates in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        choices=["train", "dev"],
        required=True,
        help="Dataset type (train|dev)"
    )
    
    parser.add_argument(
        "-e", "--evidence",
        type=str,
        choices=["gold", "hero", "infact", "no"],
        required=True,
        help="Evidence type (gold|hero|infact|no)"
    )
    
    parser.add_argument(
        "-n", "--num-items",
        type=int,
        default=0,
        help="Number of items to process (0 means all)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/results",
        help="Output directory"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    parser.add_argument(
        "--debate-model",
        type=str,
        help="Debate model name (overrides config file)"
    )
    
    parser.add_argument(
        "--judge-model", 
        type=str,
        help="Judge model name (overrides config file)"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes"
    )
    
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        default=True,
        help="Skip already processed claims"
    )
    
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Start processing from specified claim ID"
    )
    
    return parser.parse_args()


def resolve_file_paths(args):
    """Resolve data and evidence paths based on dataset and evidence parameters"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config.global_config import global_config
    
    AVERITEC_DATA_DIR = global_config.averitec_data_dir
    HERO_DATA_DIR = global_config.hero_data_dir

    if args.dataset == "train":
        args.data_path = f"{AVERITEC_DATA_DIR}/train.json"
        hero_path = f"{HERO_DATA_DIR}/train_veracity_prediction_8b.json"
        infact_path = f"{HERO_DATA_DIR}/extracted_infact_evidence.json"
    else:
        args.data_path = f"{AVERITEC_DATA_DIR}/dev.json"
        hero_path = f"{HERO_DATA_DIR}/dev_veracity_prediction_8b.json"
        infact_path= f"{HERO_DATA_DIR}/extracted_infact_evidence.json"

    args.evidence_path = None
    if args.evidence == "gold":
        args.evidence_path = args.data_path
    elif args.evidence == "hero":
        args.evidence_path = hero_path
    elif args.evidence == "infact":
        args.evidence_path = infact_path
    elif args.evidence == "no":
        args.evidence_path = None

    return args


def load_claims_from_files(args) -> list:
    """Load claim list from files based on data_path and evidence_path"""
    base_data = FileUtils.load_json(args.data_path)
    if not isinstance(base_data, list):
        base_data = [base_data]
    
    for i, item in enumerate(base_data):
        if "claim_id" not in item:
            item["claim_id"] = i

    input_data = None
    if args.evidence == "gold":
        input_data = base_data
        for item in input_data:
            if "questions" in item:
                evidence_list = []
                for q in item["questions"]:
                    for answer in q.get("answers", []):
                        evidence_list.append({
                            "question": q["question"],
                            "answer": answer.get("answer", ""),
                            "url": answer.get("source_url", "")
                        })
                if evidence_list:
                    item["evidence"] = evidence_list
    elif args.evidence in ("hero", "infact"):
        input_data = FileUtils.load_json(args.evidence_path)
        if not isinstance(input_data, list):
            input_data = [input_data]
        averitec_map = {item["claim_id"]: item for item in base_data}
        for item in input_data:
            claim_id = item.get("claim_id")
            if claim_id is not None and claim_id in averitec_map:
                for key, value in averitec_map[claim_id].items():
                    if key not in item:
                        item[key] = value
            elif claim_id is None:
                input_index = input_data.index(item)
                if input_index < len(base_data):
                    for key, value in base_data[input_index].items():
                        if key not in item:
                            item[key] = value
    elif args.evidence in ("no"):
        input_data = base_data
        for item in input_data:
            if "evidence" in item:
                del item["evidence"]
            if "questions" in item:
                del item["questions"]

    for item in input_data:
        if "pred_label" in item or "debate_answer" in item:
            original_label = item.get("pred_label", "") or item.get("debate_answer", "")
            item["pred_label"] = normalize_label(original_label)

    if args.num_items > 0:
        input_data = input_data[:args.num_items]

    if args.resume_from is not None:
        input_data = [item for item in input_data if item.get("claim_id", 0) >= args.resume_from]

    claims = []
    for item in input_data:
        try:
            claim = Claim.from_dict(item)
            claims.append(claim)
        except Exception:
            continue

    return claims


def process_single_claim(claim_data):
    """Worker function for processing single claim (for multiprocessing)"""
    claim, config_dict, output_dir, verbose = claim_data
    
    config = ConfigManager.from_dict(config_dict)
    
    openai_client = OpenAIClient(
        api_key=config.get("api_settings.openai_api_key"),
        base_url=config.get("api_settings.openai_base_url"),
        llama_api_key=config.get("api_settings.llama_api_key"),
        llama_base_url=config.get("api_settings.llama_base_url"),
        qwen_api_key=config.get("api_settings.qwen_api_key"),
        qwen_base_url=config.get("api_settings.qwen_base_url"),
        sleep_time=config.get("api_settings.sleep_time", 0)
    )
    
    debate = Debate(
        config=config,
        openai_client=openai_client,
        output_dir=output_dir,
        verbose=verbose
    )
    
    config_file_path = os.path.join(output_dir, f"{claim.claim_id}-config.json")
    config.save_config(config_file_path)
    
    result = debate.run_debate(claim)
    
    if not result.success:
        error_msg = result.metadata.get("error", "Unknown error")
        raise Exception(f"Failed to process claim {claim.claim_id}: {error_msg}")
    
    debate.save_result(claim, exclude_keys=["prompts"])
    
    return {
        "claim_id": claim.claim_id,
        "success": result.success,
        "verdict": result.verdict,
        "evidence_count": len(claim.evidence_set) if claim.has_evidence() else 0,
        "error": None
    }


def main():
    """Main function"""
    args = parse_args()
    
    try:
        args = resolve_file_paths(args)
        
        config = ConfigManager(args.config)
        
        if args.debate_model:
            config.set("models.debate_model", args.debate_model)
        if args.judge_model:
            config.set("models.judge_model", args.judge_model)
        if args.verbose:
            config.set("debate_settings.verbose", True)
        
        run_batch_claims(args, config)
        
    except Exception as e:
        sys.exit(1)


def run_batch_claims(args, config):
    """Run batch claim processing"""
    claims = load_claims_from_files(args)
    
    config_name = Path(args.config).stem
    output_dir = FileUtils.create_output_dir(
        base_dir=args.output,
        debate_model=args.debate_model or config.get("models.debate_model"),
        judge_model=args.judge_model or config.get("models.judge_model"),
        config_name=config_name,
        is_train=(args.dataset == "train"),
        evidence_type=args.evidence,
    )
    
    if args.skip_processed:
        processed_claims = FileUtils.get_processed_claims(output_dir)
        if processed_claims:
            claims = [claim for claim in claims if claim.claim_id not in processed_claims]
    
    if not claims:
        return
    
    config_dict = config.to_dict()
    claim_data_list = [
        (claim, config_dict, output_dir, args.verbose) 
        for claim in claims
    ]
    
    evidence_lengths = []
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    if args.workers == 1:
        for claim_data in tqdm(claim_data_list, desc="Processing claims"):
            result = process_single_claim(claim_data)
            
            evidence_lengths.append(result["evidence_count"])
            if result["success"]:
                success_count += 1
            else:
                error_count += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_claim = {
                executor.submit(process_single_claim, claim_data): claim_data[0].claim_id
                for claim_data in claim_data_list
            }
            
            for future in tqdm(as_completed(future_to_claim), total=len(claim_data_list), desc="Processing claims"):
                claim_id = future_to_claim[future]
                result = future.result()
                evidence_lengths.append(result["evidence_count"])
                
                if result["success"]:
                    success_count += 1
                else:
                    error_count += 1


if __name__ == "__main__":
    main()
