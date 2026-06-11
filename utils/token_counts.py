"""
uv run token-counts [experiment]

Generates the workload for an experiment and prints token distribution stats
per profile — prompt tokens, max_tokens budget, system prompt length.
"""

import sys

import tiktoken

from factory.config import load
from factory.workload import generate, _SYSTEM_CHAT, _SYSTEM_CODE


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    CFG = load(exp)
    requests = generate(CFG)

    enc = tiktoken.get_encoding("o200k_base")

    print(f"\nexperiment: {CFG.exp}")
    print(f"system prompts:")
    print(f"  chat : {len(enc.encode(_SYSTEM_CHAT))} tok")
    print(f"  code : {len(enc.encode(_SYSTEM_CODE))} tok")

    for p in ("siso", "silo", "liso", "lilo"):
        pr = [r for r in requests if r.profile == p]
        if not pr:
            continue
        user_lens = [len(enc.encode(r.user)) for r in pr]
        max_toks = [r.max_tokens for r in pr]
        print(f"\n[{p}] n={len(pr)}")
        print(
            f"  user prompt tokens : min={min(user_lens)}  max={max(user_lens)}  mean={sum(user_lens)//len(user_lens)}"
        )
        print(f"  max_tokens budget  : min={min(max_toks)}  max={max(max_toks)}")


if __name__ == "__main__":
    main()
