# Kaggle rehearsal: what we expect to see (written 1 Nov 2026, 03:15 IST, before any result)

Final settings (Qwen3-0.6B, 1,536 tokens, lr 1e-4, Dr. GRPO, 8x8 per step), seed 100, 100 steps,
Kaggle T4 (fp16). Not reported as results and not pooled with the Modal runs (Amendment 6).

Setup checks (must all pass before Modal):
- every run finishes without errors; the kill-and-resume test logs each step exactly once
- answers cut off at 1,536 tokens: under 5% of answers after step 10
- `strict` does not collapse (Amendment 3b rule)
- samples.jsonl, log.jsonl and the trace pages are complete and readable
- MBPP: code is extracted from most answers, the sandbox runs it, no timeouts storm

Behaviour we expect (if not seen, we look at why before spending, and say so in the paper):
- `strict`: true accuracy flat or rising, fp = 0
- `substring_all`: fp rises (the positive control for cheating)
- `substring_tail50`: little or no rise in fp
- (dropped from the trimmed rehearsal for time) `tail50_slope`: more rise than `substring_tail50` (this is the open question; one seed decides nothing)
- `tail50_density`: like `substring_tail50`
- `mbpp_base`: paid-but-wrong (passes the original tests, fails MBPP+) rises
- `mbpp_plus`: fp = 0, pass rate flat or rising
