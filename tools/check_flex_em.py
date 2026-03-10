#!/usr/bin/env python3
"""
Utility to extract flex exact match from lm_eval results JSON.

Usage:
    python tools/check_flex_em.py <results_dir> [--threshold 0.75] [--task gsm8k_cot]
"""

import argparse
import glob
import json
import os
import sys


def find_results_json(base_dir: str) -> str:
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        return None

    try:
        for name in os.listdir(base_dir):
            sub = os.path.join(base_dir, name)
            if os.path.isdir(sub) and "__" in name:
                cand = sorted(glob.glob(os.path.join(sub, "results_*.json")))
                if cand:
                    return cand[0]
    except Exception:
        pass

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "wandb"}]
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                return os.path.join(root, fn)

    return None


def extract_flex_em(results_path: str, task: str):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    task_data = results.get(task)
    if task_data is None:
        for k in results:
            if task.lower() in k.lower():
                task_data = results[k]
                break

    if task_data is None:
        for _, v in results.items():
            if isinstance(v, dict):
                task_data = v
                break

    if task_data is None:
        return None, None

    for key, val in task_data.items():
        if "exact_match" in key.lower() and "flex" in key.lower():
            return key, float(val)

    for key, val in task_data.items():
        if "exact_match" in key.lower() and not key.lower().endswith("_stderr"):
            try:
                return key, float(val)
            except Exception:
                continue

    for key, val in task_data.items():
        kl = key.lower()
        if ("acc" in kl or "match" in kl) and not kl.endswith("_stderr"):
            try:
                return key, float(val)
            except Exception:
                continue

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Check flex exact match from lm_eval results")
    parser.add_argument("results_dir", help="Directory containing results")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum score threshold")
    parser.add_argument("--task", type=str, default="gsm8k_cot", help="Task name")
    parser.add_argument("--quiet", action="store_true", help="Only output score value")
    args = parser.parse_args()

    results_path = find_results_json(args.results_dir)
    if results_path is None:
        if not args.quiet:
            print(f"[ERROR] No results_*.json found under {args.results_dir}", file=sys.stderr)
        sys.exit(2)

    key, score = extract_flex_em(results_path, args.task)
    if score is None:
        if not args.quiet:
            print(f"[ERROR] Could not extract metric from {results_path}", file=sys.stderr)
        sys.exit(2)

    if args.quiet:
        print(f"{score}")
    else:
        print(f"[OK] {key} = {score:.4f} (from {results_path})")

    if args.threshold is not None:
        if score >= args.threshold:
            if not args.quiet:
                print(f"[PASS] {score:.4f} >= {args.threshold}")
            sys.exit(0)
        if not args.quiet:
            print(f"[FAIL] {score:.4f} < {args.threshold}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
