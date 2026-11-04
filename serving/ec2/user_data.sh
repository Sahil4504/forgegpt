#!/bin/bash
# EC2 user data: turns a fresh Ubuntu 24.04 m7i-flex.large (2 vCPU, 8 GiB) into an
# OpenAI-compatible LLM server running Qwen2.5-0.5B-Instruct with llama.cpp.
# The whole box rebuilds itself from this script, so it can be terminated freely.
#
# Placeholders filled in by the CDK stack (infra/): __API_KEY__, __QUANT__
set -euxo pipefail
exec > >(tee /var/log/forgegpt-setup.log) 2>&1

LLAMA_TAG="b12052"                     # pinned llama.cpp release
QUANT="__QUANT__"                       # q4_k_m (491 MB) or q8_0 (676 MB)
HF="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main"

apt-get update
apt-get install -y build-essential cmake git curl

# Build only the two binaries we need. GGML_NATIVE tunes for this CPU (AVX-512 on m7i).
git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp /opt/llama.cpp
cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /opt/llama.cpp/build -j 2 --target llama-server llama-bench

mkdir -p /opt/models /etc/forgegpt
for q in q4_k_m q8_0; do
  curl -fL --retry 5 -o "/opt/models/qwen2.5-0.5b-instruct-$q.gguf" "$HF/qwen2.5-0.5b-instruct-$q.gguf"
done
echo "__API_KEY__" > /etc/forgegpt/api_key
chmod 600 /etc/forgegpt/api_key
echo "QUANT=$QUANT" > /etc/forgegpt/env

cat > /etc/systemd/system/llama-server.service <<'UNIT'
[Unit]
Description=llama.cpp server (Qwen2.5-0.5B-Instruct)
After=network-online.target

[Service]
EnvironmentFile=/etc/forgegpt/env
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  -m /opt/models/qwen2.5-0.5b-instruct-${QUANT}.gguf \
  --host 0.0.0.0 --port 8080 --threads 2 --ctx-size 4096 --parallel 4 \
  --api-key-file /etc/forgegpt/api_key --metrics
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now llama-server
