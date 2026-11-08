#!/bin/bash
# Runs every LLM-judge referee on one GPU machine, in priority order (pre-registered judges first,
# the larger gpt-oss-120b last as an exploratory extra). Expects /workspace/fdr (code) and
# /workspace/judge_inputs.jsonl. Progress in /workspace/STATUS, outputs in /workspace/judged/.
cd /workspace/fdr
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_ENABLE_HF_TRANSFER=1
echo "install start $(date -u)" >> /workspace/STATUS
{ pip install -q uv && uv pip install --system "vllm==0.10.2" "transformers==4.57.1" hf_transfer; } > /workspace/pip.log 2>&1 \
  || { echo "PIPFAILED $(date -u)" >> /workspace/STATUS; exit 1; }
pip freeze > /workspace/judged_pip_freeze.txt
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B'); t.all_special_tokens_extended; import vllm" > /workspace/judged_check.log 2>&1 \
  || { echo "PIPFAILED version check $(date -u)" >> /workspace/STATUS; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /workspace/gpu.txt
echo "install done $(date -u)" >> /workspace/STATUS
mkdir -p /workspace/judged
for job in "compassverifier-3b compass" "qwen3-8b sighted" "qwen3-8b blind" \
           "gpt-oss-20b sighted" "gpt-oss-20b blind" "gpt-oss-120b sighted"; do
  set -- $job
  echo "judge $1 $2 start $(date -u)" >> /workspace/STATUS
  python fdrl/judge.py run --inputs /workspace/judge_inputs.jsonl --judge $1 --mode $2 --out /workspace/judged \
    > /workspace/judged/$1_$2.log 2>&1 && echo "judge $1 $2 done $(date -u)" >> /workspace/STATUS \
    || echo "judge $1 $2 FAILED $(date -u)" >> /workspace/STATUS
  rm -rf /root/.cache/huggingface/hub/models--*  # free disk between models
done
echo "ALLDONE $(date -u)" >> /workspace/STATUS
