#!/usr/bin/env python
"""Compute EVT-Bench metrics (SR, TR, CR) from evaluation results."""

import os
import json
import argparse
from pathlib import Path


def compute_metrics(eval_dir: str):
    """Aggregate metrics from all episode JSON files."""
    eval_path = Path(eval_dir)
    
    success_list = []
    following_rate_list = []
    collision_list = []
    
    # Find all episode result JSONs (exclude *_info.json)
    json_files = list(eval_path.rglob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith("_info.json")]
    
    if not json_files:
        print(f"No result files found in {eval_dir}")
        return
    
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        
        # Handle both float and bool success values
        success = float(data.get("success", 0))
        following_rate = float(data.get("following_rate", 0))
        collision = float(data.get("collision", 0))
        
        success_list.append(success)
        following_rate_list.append(following_rate)
        collision_list.append(collision)
    
    n = len(success_list)
    sr = sum(success_list) / n * 100
    tr = sum(following_rate_list) / n * 100
    cr = sum(collision_list) / n * 100
    
    print(f"{'='*50}")
    print(f"Evaluation Results: {eval_dir}")
    print(f"{'='*50}")
    print(f"Episodes evaluated: {n}")
    print(f"")
    print(f"  SR (Success Rate) ↑:    {sr:.1f}%")
    print(f"  TR (Tracking Rate) ↑:   {tr:.1f}%")
    print(f"  CR (Collision Rate) ↓:  {cr:.1f}%")
    print(f"{'='*50}")
    print(f"Table format: {sr:.1f} / {tr:.1f} / {cr:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="sim_data/eval/stt", 
                        help="Path to evaluation results directory")
    args = parser.parse_args()
    compute_metrics(args.eval_dir)