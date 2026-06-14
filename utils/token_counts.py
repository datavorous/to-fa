"""
uv run token-counts [experiment]

Generates the workload for an experiment and prints per-request token counts
plus per-profile distribution stats — prompt tokens, max_tokens budget,
system prompt length.
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

    sys_chat_tok = len(enc.encode(_SYSTEM_CHAT))
    sys_code_tok = len(enc.encode(_SYSTEM_CODE))

    print(f"\nexperiment : {CFG.exp}   total requests : {len(requests)}")
    print(f"system prompts  chat={sys_chat_tok} tok   code={sys_code_tok} tok")

    for p in ("siso", "silo", "liso", "lilo"):
        pr = [r for r in requests if r.profile == p]
        if not pr:
            continue
        user_lens = [len(enc.encode(r.user)) for r in pr]
        max_toks = [r.max_tokens for r in pr]

        print(f"\n{'─'*64}")
        print(f"[{p.upper()}]  n={len(pr)}")
        print(f"  {'id':<18}  {'user tok':>8}  {'max_tok':>8}  {'total input':>11}")
        print(f"  {'─'*54}")

        for r, utok in zip(pr, user_lens):
            sys_tok = len(enc.encode(r.system))
            total_input = sys_tok + utok
            print(f"  {r.id:<18}  {utok:>8}  {r.max_tokens:>8}  {total_input:>11}")

        print(f"  {'─'*54}")
        print(
            f"  user prompt : min={min(user_lens)}  max={max(user_lens)}  mean={sum(user_lens)//len(user_lens)}"
        )
        print(f"  max_tokens  : min={min(max_toks)}  max={max(max_toks)}")


if __name__ == "__main__":
    main()
