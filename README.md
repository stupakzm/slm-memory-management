# slm-memory-management

A grounded answer engine for the Linux documentation **installed on this machine**,
built to run entirely locally on a 1.5–4B model, whose defining behaviour is that
it says *"I don't know"* rather than inventing a flag.

## Why this and not a cloud model

A cloud LLM's failure mode on `how do I exclude files from tar` is a plausible flag
that does not exist in your version of GNU tar. This system's failure mode is
silence. For system-administration questions, where a wrong flag is executed rather
than read, that is a better product — not a compromise.

## Why this and not Khoj / RAGFlow / GraphRAG

Those are corpus-agnostic, so they discard the structure that makes man pages
tractable. Man pages arrive with their hierarchy and their cross-reference graph
already authored by humans:

- `NAME / SYNOPSIS / DESCRIPTION / OPTIONS / SEE ALSO` is a document→section tree,
  parsed rather than inferred — costing zero LLM tokens.
- `SEE ALSO` is a curated edge list, the thing graph-RAG systems burn a 7B model to
  approximate.

This sidesteps the load-bearing finding in `docs/research-briefing.html`: sub-7B
models cannot build a knowledge graph, and fail hard rather than gracefully.
Nothing here asks them to.

## Status

| Phase | What | State |
|---|---|---|
| Corpus | Structured extraction from installed man pages | **done** — 4,114 pages, 38,915 sections, 15,040 cross-references |
| 0 | Evaluation set, before any system exists | **done** — 166 questions, 24% unanswerable, all verified |
| 1 | Deliberately boring flat baseline | **done** — recall@5 63.5%, answer 47.6%, prefix ablation run; see `docs/phase1-results.md` |
| 2 | Precision layer: BM25 + reranker + abstention gate | next |
| 3 | Hierarchy, only if phase 2 leaves a gap | |
| 4 | Gated tool use | |
| 5 | User-fed knowledge loop | |

## From a fresh clone

Needs Debian-ish Linux with man pages installed, Python 3.11+, and — for the model
tier — cmake plus the CUDA toolkit.

```bash
python3 -m venv --copies .venv && .venv/bin/pip install -r requirements.txt

# 1. build llama.cpp (once), outside the repo
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/opt/llama.cpp
cmake -B ~/opt/llama.cpp/build -S ~/opt/llama.cpp -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build ~/opt/llama.cpp/build -j"$(nproc)" \
      --target llama-server llama-cli llama-embedding llama-bench

# 2. models (~3.8 GB) and corpus (~30 s)
.venv/bin/python scripts/fetch_models.py
.venv/bin/python scripts/extract_man.py --sections 1,5,7,8
.venv/bin/python scripts/resolve_gold.py          # eval set must validate clean

# 3. index (~42 min on an RTX 3060)
./scripts/servers.sh start embedder
.venv/bin/python scripts/build_index.py --out data/index/phase1-prefix.db

# 4. ask it something
./scripts/ask.py "how do I exclude files listed in a text file from a tar archive"
```

Only `sqlite-vec` is a third-party Python dependency; everything else is stdlib.

### Reproducing the phase 1 numbers

The embedder and the 4B generator together need ~5.5 GB of VRAM, so answer
evaluation runs in two stages with one model resident at a time:

```bash
./scripts/servers.sh start embedder
.venv/bin/python scripts/eval_retrieval.py --db data/index/phase1-prefix.db --name phase1-prefix
.venv/bin/python scripts/eval_answers.py  --db data/index/phase1-prefix.db --name phase1-prefix --stage retrieve
./scripts/servers.sh stop && ./scripts/servers.sh start generator
.venv/bin/python scripts/eval_answers.py --name phase1-prefix --stage generate
./scripts/servers.sh stop
```

## Corpus extraction

```bash
.venv/bin/python scripts/extract_man.py --sections 1,5,7,8   # ~27 s, 12 threads
.venv/bin/python scripts/resolve_gold.py                     # validate the eval set
.venv/bin/python scripts/corpus_grep.py --doc tar.1 -- '--exclude-from'
```

`extract_man.py` writes `data/corpus/man.jsonl` (one record per page) and
`data/corpus/man_stats.json`. Sections 1, 5, 7 and 8 are indexed; man2 and man3 are
C programming reference and are deliberately excluded.

## Layout

```
src/smm/corpus/manpages.py   discover / render / parse man pages
scripts/extract_man.py       corpus extraction driver
data/corpus/                 extracted JSONL (gitignored)
scripts/corpus_grep.py       search the extracted corpus
scripts/resolve_gold.py      eval-set validator: no gold by assertion, no leaked answers
data/eval/questions.jsonl    phase 0 evaluation set (version-controlled)
docs/research-briefing.html  the research this design follows
```
