"""CLI entry points (used by ``project.scripts`` in pyproject.toml).

Implementations are dispatched to scripts/ and module-level functions in
``training``, ``eval``, ``attacker``. Stage 3.1 provides only stubs; full
implementations are added in Stages 3.4-3.6.
"""
from __future__ import annotations

import argparse
import sys


def train() -> int:
    parser = argparse.ArgumentParser(prog="diploma-train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"[train] model={args.model} train={args.train} seed={args.seed}")
    print("[train] full implementation arrives in Stage 3.4")
    return 0


def evaluate() -> int:
    parser = argparse.ArgumentParser(prog="diploma-eval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval", required=True)
    args = parser.parse_args()
    print(f"[eval] model={args.model} eval={args.eval}")
    print("[eval] full implementation arrives in Stage 3.4")
    return 0


def run_attacker() -> int:
    parser = argparse.ArgumentParser(prog="diploma-attack")
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    print(f"[attack] policy={args.policy}")
    print("[attack] full implementation arrives in Stage 3.5")
    return 0


if __name__ == "__main__":
    sys.exit(train())
