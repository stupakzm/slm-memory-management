#!/usr/bin/env bash
# Start/stop the local llama-server instances this project talks to.
#
#   ./scripts/servers.sh start serve         # all three, for asking questions
#   ./scripts/servers.sh start embedder      # indexing profile, big batches
#   ./scripts/servers.sh start generator     # port 8080
#   ./scripts/servers.sh status
#   ./scripts/servers.sh stop
#
# Phase 2 answers a question with three models, and on a 6 GB card they only fit
# together if the two small ones are sized for queries rather than for indexing.
# `start embedder` allocates a 600 MB compute buffer for 4096-token batches, which
# is right for a 42-minute index build and OOMs the moment the generator is also
# resident. `start serve` is the same three models with query-sized batches:
# embedder ~0.8 GB + reranker ~0.8 GB + generator ~3.7 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${LLAMA_CPP:-$HOME/opt/llama.cpp/build/bin}"
MODELS="$ROOT/models"
RUN="$ROOT/.run"
mkdir -p "$RUN"

start_embedder() {
  "$LLAMA/llama-server" \
    -m "$MODELS/Qwen3-Embedding-0.6B-Q8_0.gguf" \
    --embedding --pooling last -c 8192 -ngl 99 \
    -b 4096 -ub 1024 --parallel 4 \
    --host 127.0.0.1 --port 8081 \
    > "$RUN/embedder.log" 2>&1 &
  echo $! > "$RUN/embedder.pid"
  echo "embedder starting (pid $(cat "$RUN/embedder.pid")) -> $RUN/embedder.log"
}

# Five 1,000-char extracts plus the question and the answer come to ~2,200 tokens,
# so the serve profile drops the context to 4096 and gives the 1.1 GB of KV cache
# that 8192 was reserving back to the other two models.
start_generator() {
  local ctx=${1:-8192}
  "$LLAMA/llama-server" \
    -m "$MODELS/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
    -c "$ctx" -ngl 99 --temp 0.0 \
    --host 127.0.0.1 --port 8080 \
    > "$RUN/generator.log" 2>&1 &
  echo $! > "$RUN/generator.pid"
  echo "generator starting (pid $(cat "$RUN/generator.pid")) -> $RUN/generator.log"
}

# A query-plus-chunk pair runs to ~550 tokens, so the physical batch has to clear
# that or llama-server rejects the request outright rather than splitting it.
start_reranker() {
  local ctx=${1:-8192} batch=${2:-2048} par=${3:-4}
  "$LLAMA/llama-server" \
    -m "$MODELS/qwen3-reranker-0.6b-q8_0.gguf" \
    --reranking -c "$ctx" -ngl 99 \
    -b "$batch" -ub "$batch" --parallel "$par" \
    --host 127.0.0.1 --port 8082 \
    > "$RUN/reranker.log" 2>&1 &
  echo $! > "$RUN/reranker.pid"
  echo "reranker starting (pid $(cat "$RUN/reranker.pid")) -> $RUN/reranker.log"
}

# Query-sized: an instruction-wrapped question is ~50 tokens, so 512 of context is
# already generous and the compute buffer shrinks with it.
start_embedder_lean() {
  "$LLAMA/llama-server" \
    -m "$MODELS/Qwen3-Embedding-0.6B-Q8_0.gguf" \
    --embedding --pooling last -c 512 -ngl 99 \
    -b 512 -ub 512 --parallel 1 \
    --host 127.0.0.1 --port 8081 \
    > "$RUN/embedder.log" 2>&1 &
  echo $! > "$RUN/embedder.pid"
  echo "embedder starting (pid $(cat "$RUN/embedder.pid")) -> $RUN/embedder.log"
}

wait_ready() {
  local port=$1 name=$2
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      echo "$name ready on $port"; return 0
    fi
    sleep 1
  done
  echo "$name did not come up; see $RUN/$name.log" >&2
  tail -20 "$RUN/$name.log" >&2 || true
  return 1
}

case "${1:-status}" in
  start)
    case "${2:-embedder}" in
      embedder)  start_embedder;  wait_ready 8081 embedder ;;
      generator) start_generator; wait_ready 8080 generator ;;
      reranker)  start_reranker;  wait_ready 8082 reranker ;;
      serve)
        start_embedder_lean; wait_ready 8081 embedder
        start_reranker 1024 768 1; wait_ready 8082 reranker
        start_generator 4096; wait_ready 8080 generator ;;
      *) echo "unknown: $2" >&2; exit 1 ;;
    esac ;;
  stop)
    for f in "$RUN"/*.pid; do
      [ -e "$f" ] || continue
      pid=$(cat "$f"); kill "$pid" 2>/dev/null && echo "stopped $(basename "$f" .pid) ($pid)"
      rm -f "$f"
    done ;;
  status)
    for p in 8080:generator 8081:embedder 8082:reranker; do
      port=${p%%:*}; name=${p##*:}
      if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        echo "  $name  UP    :$port"
      else
        echo "  $name  down  :$port"
      fi
    done
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader ;;
  *) echo "usage: $0 {start {serve|embedder|generator|reranker}|stop|status}" >&2; exit 1 ;;
esac
