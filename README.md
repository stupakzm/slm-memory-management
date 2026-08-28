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

The other half of the design is that abstention is architectural. A score gate
refuses before the generator is invoked, so on thin evidence there is nothing to
speculate with, and a GBNF grammar makes an uncited claim structurally impossible.
Phase 1 measured what asking nicely in a prompt buys instead: fabricated citations
like `NULL [3]`, pointing at an extract that says nothing of the kind.

## Status

| Phase | What | State |
|---|---|---|
| Corpus | Structured extraction from installed man pages | **done** — 4,147 pages; a phase 4 fix recovered cross-reference tags on 1,084 of them |
| 0 | Evaluation set, before any system exists | **done** — 166 questions, 24% unanswerable, all verified |
| 1 | Deliberately boring flat baseline | **done** — recall@5 63.5%, answer 47.6%, prefix ablation run; see `docs/phase1-results.md` |
| 2 | Precision layer: reranker + abstention gate + GBNF citations | **done** — recall@5 71.4%, abstention 92.5%, uncited claims 0%; BM25 measured and rejected; see `docs/phase2-results.md` |
| 3 | Structure-aware chunking | **done, reverted** — +8 questions, −8 questions, net zero; and cleaner chunks made the abstention gate *worse*; see `docs/phase3-results.md` |
| 4 | Gated tool use | **done** — zero unconfirmed executions across 480 requests; GBNF takes valid calls from 17.5% to 100%; see `docs/phase4-results.md` |
| 5 | User-fed knowledge loop | next |

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
.venv/bin/python scripts/build_lexical.py --from data/index/phase1-prefix.db --out data/index/phase2.db
./scripts/servers.sh stop && ./scripts/servers.sh start serve
.venv/bin/python scripts/ask.py "how do I exclude files listed in a text file from a tar archive"
.venv/bin/python scripts/ask.py --act "make a gzip-compressed archive of the reports folder"
```

`--act` is tool mode: the system answers, proposes a command, opens a manual page, or
refuses. A proposed command is printed with its risk and **never run** — execution is
opt-in per tool and only a validated, read-only manual page qualifies.

Phase 2 answers a question with three models at once, so `servers.sh start serve`
runs them with query-sized batches — the indexing profile reserves a 600 MB compute
buffer and will not fit beside the generator on a 6 GB card.

Only `sqlite-vec` is a third-party Python dependency; everything else is stdlib.

### Reproducing the numbers

Evaluation runs in two stages with one set of models resident at a time, because the
embedder, the reranker and the 4B generator do not all fit at eval batch sizes.

```bash
# retrieval, and the ablations behind every phase 2 claim
./scripts/servers.sh start embedder && ./scripts/servers.sh start reranker
.venv/bin/python scripts/eval_retrieval.py --db data/index/phase2.db --mode dense --name phase2-dense
.venv/bin/python scripts/eval_retrieval.py --db data/index/phase2.db --mode bm25  --name phase2-bm25
.venv/bin/python scripts/eval_retrieval.py --db data/index/phase2.db --mode dense --rerank --name phase2-dense-rerank

# answers: retrieve once, then replay generation variants against the same cache
.venv/bin/python scripts/eval_answers.py --stage retrieve --mode dense --rerank --name phase2-full
./scripts/servers.sh stop && ./scripts/servers.sh start generator
.venv/bin/python scripts/eval_answers.py --stage generate --name phase2-full --gate 0.65 --grammar
./scripts/servers.sh stop
```

The gate threshold is not a taste decision. `sweep_gate.py --answers phase2-rerank`
replays every threshold against one ungated generation run — the gate only chooses
whether to invoke the model, never what it writes — and reports accuracy, abstention
recall and unsupported answers at each. 0.65 is the point it picks.

`compare_runs.py` reports fixed/broken by question id for any two retrieval runs.

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
src/smm/structure.py         split a section into the entries a reader sees in it
src/smm/tools.py             tool schemas, GBNF per schema, validate-before-execute
src/smm/chunk.py             flat windows (default) and the structure-aware chunker
src/smm/retrieve.py          the pipeline: candidates -> fuse -> rerank -> gate
src/smm/rerank.py            cross-encoder client
src/smm/lexical.py           BM25 on FTS5 - measured, and off by default
src/smm/grammar.py           GBNF citation grammar and the check it enables
scripts/extract_man.py       corpus extraction driver
scripts/build_lexical.py     add the BM25 half to an existing dense index
scripts/chunk_stats.py       chunking cost and ceiling, before spending GPU time
scripts/corpus_grep.py       search the extracted corpus
scripts/resolve_gold.py      eval-set validator: no gold by assertion, no leaked answers
scripts/resolve_tools.py     the same rule for the tool eval set
scripts/eval_tools.py        tool-call accuracy, over-action, and the execution gate
scripts/sweep_gate.py        pick the abstention threshold on the labelled set
scripts/compare_runs.py      per-question diff between two runs
data/eval/questions.jsonl    phase 0 evaluation set (version-controlled)
data/eval/tool_questions.jsonl  phase 4 tool eval set (version-controlled)
tests/test_tools.py          asserts the execution gate directly
docs/research-briefing.html  the research this design follows
```
