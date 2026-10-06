# Training the reader without a rented GPU

A runbook for `scripts/train_reader.py` on hardware that has no bf16 or little memory: the Colab
free tier, Kaggle notebooks, and the local RTX 3060 Laptop (6 GB). Read the last section before you
spend hours on it: the training half of the script has never been run end to end.

## 1. What is being trained, and why

The 4B reader (Qwen3-4B-Instruct-2507) gets a LoRA fine-tune on RAFT-style chat data: each example
is a question, five extracts (the gold chunk among dense-search distractors, or five distractors
with the gold removed), and a short cited answer, or "I don't know" when the gold is removed. The
loss is over the answer tokens only. The reasoning is `docs/phase16-results.md` idea 15 and
`todays-plan.md` section D, item 15: reading has been the bottleneck since phase 8 and prompting
has gone as far as it can. Nothing here is measured. The result is judged only by the same
pre-registered generation arm every other reader arm goes through (section 5).

`train_reader.py` has three switches for small hardware. Defaults are the old behaviour (bf16,
16-bit base, merge in place).

| flag | what it does |
|---|---|
| `--precision fp16` | fp16 base and a loss scaler (`torch.cuda.amp.GradScaler`). For GPUs without bf16: T4, P100. The default `bf16` is refused on such a GPU, with a message that names this flag. |
| `--load-in-4bit` | QLoRA: the base is loaded as 4-bit NF4 (double quantisation, compute dtype = the chosen precision) with the same hand-written LoRA on top. Needs `--device cuda`. |
| `--merge-device cpu` | merge the adapter into a float16 copy of the base loaded on the CPU. Always the case under `--load-in-4bit`, because the adapter is never merged into quantised weights. |

## 2. Data

Build the training file where the local GPU and the 30B teacher are (`plan` needs the embedder server, `teach` the 30B server, `assemble`
neither):

```bash
.venv/bin/python scripts/build_raft_data.py plan     --work data/raft --questions data/index/qvec-emacs.json \
    --questions data/index/qvec-linux.json --eval data/eval/questions.jsonl \
    --eval data/eval/emacs_questions.jsonl --eval data/eval/variations_typo.jsonl \
    --eval data/eval/variations_man.jsonl --eval data/eval/variations_emacs.jsonl --n 2000
.venv/bin/python scripts/build_raft_data.py teach    --work data/raft --gen-url <the 30B llama-server URL>
.venv/bin/python scripts/build_raft_data.py assemble --work data/raft
```

The two `--questions` files are the question caches the R13 embedder training used (`data/index/qvec-emacs.json`,
`data/index/qvec-linux.json`); the five `--eval` files are the pool's eval sets, so no page that any of them
treats as gold is trained on. The 30B teacher is local and takes about 8 s
per example (an estimate from `todays-plan.md`, not a fresh measurement). Size the pilot at 1,500 to
2,000 examples: that is about 3.5 to 4.5 hours of local GPU for `teach`. The output is
`data/raft/train.jsonl`. Upload it (Google Drive for Colab, a private Kaggle dataset for Kaggle).
The base model directory is not in the repo either; each recipe downloads it.

## 3. Recipes

All three use the same command shape. Paths may be absolute: `--data`, `--out` and `--model-dir`
are joined to the repo root, and an absolute path stays as it is.

**First run on any machine: 5 to 10 steps on a 20-example file.** Cut 20 lines
(`head -n 20 train.jsonl > tiny.jsonl`) and run with `--grad-accum 2 --epochs 1 --gen-dev 0
--eval-every 0 --dev-frac 0.2`. A tiny file has few dev examples, so this proves that the model
loads, one step runs, `adapter.pt` is written and `merged/` appears. It does not prove that the
training is good. Read the step time off the log, then decide the real plan from it.

How a long run is kept alive: `--resume --stop-after-s N` finishes the step that crosses N seconds,
saves, and exits 0. Exit 0 is also what a finished run returns, so the exit code alone does not
tell the two apart. Read `train_log.json` in `--out`: `"status": "paused"` means run again with
`--resume`, `"status": "complete"` means stop (the last lines then say `merged N modules`). The loop
below does that. A checkpoint is also written every `--checkpoint-every` steps (default 100), which
is what saves a session that is killed outright.

### 3a. Colab free (one T4, 16 GB, no bf16)

```bash
# a GPU runtime; then in a cell:
from google.colab import drive; drive.mount('/content/drive')
!git clone <your repo URL> /content/smm && cd /content/smm   # or upload a zip of scripts/, src/
# the pinned environment: see section 4
!huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /content/smm/models/hf/Qwen3-4B-Instruct-2507
```

```bash
cd /content/smm
OUT=/content/drive/MyDrive/smm/reader-r1
ARGS="--data /content/drive/MyDrive/smm/train.jsonl --out $OUT --precision fp16 --max-len 2048 --grad-accum 16"
python scripts/train_reader.py $ARGS --stop-after-s 10000
until grep -q '"status": "complete"' $OUT/train_log.json; do
  python scripts/train_reader.py $ARGS --resume --stop-after-s 10000
done
```

Put `--out` on Drive, so a recycled runtime keeps `adapter.pt`, `ckpt.pt` and `resume_state.json`. If
the first step runs out of GPU memory, add `--load-in-4bit` to `ARGS` (and reduce `--max-len`
further if it still does). Under `--load-in-4bit` the final merge reloads the base in float16 on the
CPU, which needs roughly 8 GB of system RAM for a 4B (2 bytes per weight; arithmetic, not a
measurement); if the runtime has less, the run still leaves `adapter.pt`, and you can run
`--merge-only --merge-device cpu` on a machine that has the RAM. Start a rerun in a new cell
after the session is reclaimed: the first command refuses to start over an existing `--out` (it asks
for `--resume`), which is the intended protection.

### 3b. Kaggle (P100 or 2x T4, 16 GB each, no bf16; about 30 GPU hours a week)

Use one GPU: the script does not use two. Same as Colab, but with the Kaggle paths. Turn Internet on
in the notebook settings to download the base model, and add `train.jsonl` as an input dataset.

```bash
cd /kaggle/working/smm
OUT=/kaggle/working/reader-r1
ARGS="--data /kaggle/input/smm-raft/train.jsonl --out $OUT --precision fp16 --max-len 2048 --grad-accum 16"
python scripts/train_reader.py $ARGS --stop-after-s 10000
until grep -q '"status": "complete"' $OUT/train_log.json; do
  python scripts/train_reader.py $ARGS --resume --stop-after-s 10000
done
```

`/kaggle/working` is lost when the session ends unless you keep it: at the end of every session save
the output as a dataset version (Save Version, or `kaggle datasets version`), and at the start of the
next session add that dataset as an input and copy it back to `/kaggle/working/reader-r1` before the
`--resume` command. The weekly GPU-hour quota is the platform's published figure at the time this was
written; check it. Add `--load-in-4bit` as under Colab if the first step does not fit.

### 3c. Local RTX 3060 Laptop (6 GB, supports bf16)

6 GB does not hold a 4B in 16 bits, so this is QLoRA, and it is expected to be tight. Nothing else may
hold the GPU: stop every llama-server first.

```bash
./scripts/servers.sh stop
nvidia-smi          # nothing else should be using VRAM
.venv-train/bin/pip install bitsandbytes     # into .venv-train, see section 4
.venv-train/bin/python scripts/train_reader.py --data data/raft/train.jsonl --out data/reader/r1 \
    --load-in-4bit --precision bf16 --max-len 2048 --grad-accum 16 --stop-after-s 10000
```

`--precision fp16` works too and is the fallback if bf16 misbehaves. `--max-len 2048` drops examples
longer than that rather than cutting them; the log's `dropped over 2048 tokens` line says how many.
If it still runs out of memory, lower `--max-len` before anything else. Continue with `--resume` as
in the loop above (`--out data/reader/r1`). This laptop has not been used for this; expect trial and
error.

## 4. Environment

`requirements-train.txt` pins torch 2.6.0 and transformers 4.57.6 and is not edited for this. On top
of it, install:

```bash
pip install bitsandbytes          # --load-in-4bit; any release that supports your torch
pip install accelerate            # transformers needs it for device_map (4-bit); if not already there
```

Colab and Kaggle caveat: their preinstalled torch and transformers differ from the pins, and
`bitsandbytes` has to match the torch it runs under. Do not train on the preinstalled stack and hope.
Either build a fresh venv (`python3 -m venv /content/venv-train`, then the three install lines at the
top of `requirements-train.txt`), or install the pins into a separate directory with
`pip install --target /content/pylibs -r requirements-train.txt` and put it first on `PYTHONPATH`.
`gguf` is not needed for training, only for conversion (section 5), on the local machine.

## 5. After training

1. Copy `merged/` (safetensors, config, tokenizer files) back to the local machine, for example to
   `data/reader/r1/merged`.
2. Convert with llama.cpp to f16, then quantise to Q4_K_M:
   ```bash
   python ~/opt/llama.cpp/convert_hf_to_gguf.py data/reader/r1/merged --outtype f16 \
       --outfile models/reader-r1-f16.gguf
   ~/opt/llama.cpp/build/bin/llama-quantize models/reader-r1-f16.gguf models/reader-r1-Q4_K_M.gguf Q4_K_M
   ```
   (the llama.cpp path is where your checkout builds; adjust it). Put the GGUF where
   `scripts/servers.sh` looks for models.
3. Serve it as the reader: `SMM_GEN_MODEL=reader-r1-Q4_K_M.gguf SMM_GEN_PORT=8090 SMM_GEN_ARGS="--parallel 1
   --no-cache-prompt --cache-ram 0" ./scripts/servers.sh start generator`, the same setting as the
   control `p15-ctl` (docs/phase16-results.md, "Control").
4. Run the same pre-registered generation arm as every other reader arm: the control is `p15-ctl`,
   the arm reads the control's cached retrieval:
   `.venv/bin/python scripts/eval_answers.py --stage generate --cache p13-ctl --name p17-reader ...`
   with the same flags as the control run (gate 0.65, `--grammar`, the five eval files, the aliases
   file). Copy them from the control's pre-registration rather than from here. Then
   `scripts/screen_report.py` with `--control p15-ctl` and `--arm p17-reader`, scored on rules 1 to 3
   of the phase 16 decision rule (answerable net >= +8 with sign test p < 0.05; clean group net >= -2;
   abstention net >= -1), and a judged pass on the rows that differ with `scripts/judge_pack.py`.
   Write the pre-registration (what is run, which rule decides) before the generation arm, as for any
   other arm.
5. Evaluation never trains: `build_raft_data.py` excluded the eval gold docs and near-duplicate
   questions. If you change the data, rebuild through that script, never by hand.

## 6. Honest limits

- The training half of `train_reader.py` has never been run end to end. Not run on any GPU: the
  4-bit load, the fp16 loss scaler step, gradient checkpointing, the optimizer, checkpoint
  save/restore, greedy generation, the CPU merge, and `save_merged`. The tests (`tests/test_train_reader.py`)
  cover only the pure helpers. Expect a first-run bug.
- fp16 can overflow in the forward pass (large activations); the loss scaler protects the
  gradients, not the activations. If the loss is `nan` from the first step, a T4 has no other
  precision to offer: try a shorter `--max-len`, or train on the local card with bf16.
- All time estimates here are guesses. A guess for a T4: 4B in fp16 or 4-bit trains at perhaps 1 to
  1.5k tokens per second, so 2,000 examples of about 2.5k tokens (5M tokens) is on the order of 1 to 2
  hours per epoch, and the default is 2 epochs. Replace this with the first measured step time. The
  local 3060 will be slower than a T4 only if memory pressure forces it to; unknown.
- The refusal behaviour is the risk: in earlier phases arms that answered more also invented more.
  The refusal precision/recall in `dev_before.json` and `train_log.json` is the first signal, and rule 3
  decides.
- A model trained on data that the 30B wrote inherits the 30B's style and its mistakes; the
  identifier check in `assemble` limits only the worst of them.
