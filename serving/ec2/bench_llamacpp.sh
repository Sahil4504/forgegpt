#!/bin/bash
# Run on the EC2 instance (via SSM Session Manager or SSH) to measure raw llama.cpp speed
# for both quantizations, then switch the server between them.
#   sudo bash bench_llamacpp.sh            -> /opt/forgegpt-llama-bench.csv
#   sudo bash bench_llamacpp.sh use q8_0   -> restart the server with Q8_0
set -euo pipefail
if [[ "${1:-}" == "use" ]]; then
  echo "QUANT=$2" > /etc/forgegpt/env && systemctl restart llama-server && exit 0
fi
systemctl stop llama-server   # give llama-bench both vCPUs
out=/opt/forgegpt-llama-bench.csv
rm -f "$out"
for q in q4_k_m q8_0; do
  /opt/llama.cpp/build/bin/llama-bench -m "/opt/models/qwen2.5-0.5b-instruct-$q.gguf" \
    -t 2 -p 64 -n 64 -r 3 -o csv | sed "s/^/$q,/" >> "$out"
done
systemctl start llama-server
cat "$out"
