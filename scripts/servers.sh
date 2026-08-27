#!/usr/bin/env bash
# Start/stop the local llama-server instances this project talks to.
#
#   ./scripts/servers.sh start embedder      # port 8081, needed to index or query
#   ./scripts/servers.sh start generator     # port 8080, needed to answer
#   ./scripts/servers.sh status
#   ./scripts/servers.sh stop
#
# Both fit in 6 GB VRAM at once: 0.6B Q8 embedder (~0.7 GB) + 4B Q4 generator (~2.7 GB).
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

start_generator() {
  "$LLAMA/llama-server" \
    -m "$MODELS/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
    -c 8192 -ngl 99 --temp 0.0 \
    --host 127.0.0.1 --port 8080 \
    > "$RUN/generator.log" 2>&1 &
  echo $! > "$RUN/generator.pid"
  echo "generator starting (pid $(cat "$RUN/generator.pid")) -> $RUN/generator.log"
}

start_reranker() {
  "$LLAMA/llama-server" \
    -m "$MODELS/qwen3-reranker-0.6b-q8_0.gguf" \
    --reranking -c 4096 -ngl 99 \
    --host 127.0.0.1 --port 8082 \
    > "$RUN/reranker.log" 2>&1 &
  echo $! > "$RUN/reranker.pid"
  echo "reranker starting (pid $(cat "$RUN/reranker.pid")) -> $RUN/reranker.log"
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
      all) start_embedder; start_generator; wait_ready 8081 embedder; wait_ready 8080 generator ;;
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
  *) echo "usage: $0 {start {embedder|generator|reranker|all}|stop|status}" >&2; exit 1 ;;
esac
