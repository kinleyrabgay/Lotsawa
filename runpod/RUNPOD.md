# Lotsawa — RunPod training runbook

Full sequence, in order. Roughly 5–13 GPU hours and $4–12 end to end on an
RTX 4090.

---

## Step 0 — Settled: the shad is written

`normalize.py` sets `SHAD_AFTER_GA = True`, writing a shad after syllables ending
in ག/ཀ (`འདུག།`, `དོ་ག།`). Traditional Tibetan orthography omits it there; modern
Dzongkha keeps it, and that is the convention this project follows — confirmed by
the project owner. This affects **~16%** of the corpus, since ནུག, འདུག and the
question particle ག are among the most common final syllables.

It is baked into every dataset `prepare_data.py` builds, so changing it later
means rebuilding the data *and* redoing the baseline.

Sanity-check the normalizer's output before you start:

```bash
python normalize.py ../dataset.csv | head -60
```

---

## Step 1 — Create the pod

| Setting | Value |
|---|---|
| GPU | **RTX 4090 24GB** — $0.74/hr (RTX 5090 if you can get one) |
| Template | **RunPod PyTorch 2.8.0** — works for Blackwell, Ada and Ampere alike |
| Network volume | **100GB**, mounted at `/workspace` (50GB works — see below) |
| Container disk | **30GB** — the default is fine, nothing large lands here |

The 600M fine-tune needs roughly 13–16GB including optimizer states and
activations at `--max-length 128`, so 24GB fits with headroom. Rent 48GB+ when
you actually train NLLB-1.3B (Stage 3), not before.

### Choosing the GPU

Observed RunPod pricing — check the live page, these move:

| GPU | $/hr | VRAM | Gen | Notes |
|---|---|---|---|---|
| **RTX 5090** | **0.99** | 32GB | Blackwell | **Best value.** ~1.4–1.7× a 4090, so *lower total cost* despite the hourly rate. 12 vCPU / 100GB RAM. Needs a CUDA 12.8+ template |
| RTX 4090 | 0.74 | 24GB | Ada | Safe fallback — no template risk. Only 8 vCPU |
| RTX 3090 | 0.50 | 24GB | Ampere | Cheapest. ~1.4× slower than a 4090, so total cost is a wash. 32 vCPU |
| RTX PRO 4000 | 0.57 | 24GB | Blackwell | Cheaper than the 4090, same template requirement as the 5090 |
| L40S | 0.99 | 48GB | Ada | For Stage 3 / NLLB-1.3B |
| A100 SXM | 1.59 | 80GB | Ampere | For Stage 3 / NLLB-3.3B |
| RTX PRO 6000 | 2.09 | 96GB | Blackwell | Only for 3.3B |

> **Use the PyTorch 2.8.0 template regardless of card.** It covers Blackwell
> (sm_120), Ada (sm_89) and Ampere (sm_86), so you can switch GPUs without
> touching the template. An older CUDA 12.4 template has no Blackwell kernels
> and fails with `no kernel image is available for execution on the device`.
> Run the Step 3 verification before starting a long job either way.

> **The 5090 is usually out of capacity.** Consumer Blackwell is scarce on
> RunPod — the listing shows `1 max` and often reads "Instance not available".
> Do not wait on "Deploy when available"; take the 4090, which costs about $1
> more across the whole pipeline.

### Why the 5090 costs less despite the higher rate

Peak specs are ~1.9× a 4090; realistically expect 1.4–1.7× on a 600M model with
short sequences.

```
4090:  ~14h x $0.74  =  $10.4
5090:  ~9h  x $0.99  =  $9.2     <- cheaper, and finishes ~5h sooner
```

Its 32GB also removes any OOM risk at `--max-length 128`, and 12 vCPU beats the
4090 offering's 8. Take the 4090 only if you would rather not deal with template
versions.

### Storage budget

Container disk holds only the OS and pip packages, because `HF_HOME` points at
`/workspace` (Step 3). The default 30GB is plenty and it is erased when the pod
stops — nothing you care about lives there.

The network volume is the one to size. A 600M checkpoint is bigger than the
model, because it carries the optimizer state needed to resume:

```
model.safetensors    2.46 GB    615M params, fp32
optimizer.pt         4.92 GB    Adam m + v states
                   ──────────
                   ~7.4 GB per checkpoint
```

| Item | Size |
|---|---|
| HF cache for NLLB-600M | ~5 GB (ships both `.bin` and `.safetensors`) |
| Checkpoints, `--save-total-limit 2` + the exempt best | ~22 GB |
| Final saved model | 2.5 GB |
| All datasets (corpus, monolingual, prepared, synthetic) | <0.5 GB |
| **Total, one training run** | **~30 GB** |

So 50GB fits one run comfortably and **two runs not at all**. Between Step 6 and
Step 8, once the first model is safely on the Hub, drop the optimizer states —
back-translation only needs the weights:

```bash
du -sh /workspace/ckpt
rm -rf /workspace/ckpt/checkpoint-*      # keeps the final model, frees ~22GB
du -sh /workspace/ckpt
```

**Provision 100GB and skip the cleanup.** It holds both runs' checkpoints
(~60GB) at once, so you can score `ckpt` against `ckpt-bt` directly instead of
deleting one to fit the other.

Observed billing for a 4090 + 30GB container + 100GB volume:

| Line | Rate | Notes |
|---|---|---|
| GPU | $0.74/hr | only while running |
| Container disk (30GB) | $0.004/hr | only while running |
| Volume disk (100GB) | $0.014/hr | **~$10/month, always** |
| **Stopped cost** | **$0.028/hr** | **~$20/month — see below** |

Two billing traps:

- **"Stopped" is not "off."** A stopped pod keeps billing $0.028/hr forever.
  When you finish a session, **terminate** the pod — the network volume keeps
  your checkpoints, and you drop to volume-only charges.
- Volume storage bills with no pod attached at all, so **delete the volume once
  the models are on the Hub** and the project is done.

Three things that bite people:

- The **network volume must be in the same datacenter as the pod**. Pick the GPU
  and datacenter first, then create the volume there.
- Spot/interruptible saves ~50% and is safe here, because everything resumes
  from checkpoints on the network volume.
- The 4090 offering has only **8 vCPU / 31GB RAM**. Fine here, because the data
  is tokenized before the training loop. But if `nvidia-smi` shows GPU
  utilization under ~70% with `--batch 32`, lower `--dataloader-num-workers`
  from 4 to 2 before assuming the GPU is at fault.

Add these as pod **Secrets** (not in code, not in the notebook):
`HF_TOKEN` (needs *write* access to push models), and `WANDB_API_KEY` if you
want logging.

---

## Step 2 — Upload

From your Mac, in the project root:

```bash
tar czf lotsawa.tar.gz --exclude='data/dz_en_bidi' --exclude='__pycache__' runpod/ dataset.csv
runpodctl send lotsawa.tar.gz
```

That prints a one-time code. On the pod (web terminal or SSH):

```bash
cd /workspace
runpodctl receive <the-code>
tar xzf lotsawa.tar.gz && cd runpod
```

~68MB. `data/dz_en_bidi` is excluded deliberately — it rebuilds on the pod in
about a minute, and re-deriving it there proves the pipeline works on the
machine that will train.

Alternative: push the repo to GitHub and `git clone` on the pod. Better
long-term, and you should be versioning this anyway.

---

## Step 3 — Set up the environment

```bash
cd /workspace/runpod
export HF_HOME=/workspace/hf          # keep the 2.4GB model cache off the container disk
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
pip install -r requirements.txt       # deliberately does NOT install torch
huggingface-cli login                 # paste the token at the prompt
nvidia-smi                            # confirm the GPU is what you paid for
```

`requirements.txt` omits `torch` on purpose: the template ships a build compiled
against the pod's exact CUDA version, and letting pip replace it with a wheel for
a different CUDA breaks at the first kernel launch.

`hf_transfer` is in the requirements because RunPod's templates export
`HF_HUB_ENABLE_HF_TRANSFER=1` without installing the package, which makes every
Hub download fail with *"Fast download using 'hf_transfer' is enabled ... but
'hf_transfer' package is not available"*. If you hit that anyway,
`pip install hf_transfer` fixes it, or `export HF_HUB_ENABLE_HF_TRANSFER=0`
disables the flag.

Log in interactively rather than with `--token $HF_TOKEN`. RunPod Secrets are not
auto-exported as environment variables, so that variable is usually empty — and
the interactive prompt keeps the token out of your shell history. **The token
needs `Write` role**; a read token pulls the dataset fine but fails at the first
checkpoint push, hours into training. Verify with `hf auth whoami`.

**Verify the GPU actually runs a kernel before starting a multi-hour job.**
`torch.cuda.is_available()` can return `True` on a card whose architecture the
build does not support, so run a real matmul:

```bash
python check_gpu.py
```

(Use the script rather than an inline `python -c`. Long one-liners get
line-wrapped by the terminal on paste, and the injected newline produces
`IndentationError: unexpected indent`.)

Expect `sm (8, 9)` on a 4090, `sm (12, 0)` on a 5090, and `matmul ok: True`. If
the matmul raises `no kernel image is available`, the template is too old for the
card — destroy the pod and pick a newer one.

`HF_HOME` matters. The default cache lives on the container disk, which is small
and does not survive the pod. Pointing it at the network volume also means the
model downloads once across pod restarts.

---

## Step 4 — Build the dataset

```bash
python prepare_data.py --csv ../dataset.csv --out /workspace/data/dz_en_bidi
```

Expected output:

```
Loaded: {'train': 225565, 'validation': 3436, 'test': 3488}
Train: kept 219560, dropped 1457 dupes, 4548 leaked
Orthography restoration: ON
Final: {'train': 439120, 'validation': 6872, 'test': 6976}
```

439,120 = 219,560 pairs × 2 directions. If your numbers differ, stop and find out
why before training.

Use `--dataset kinleyrabgay/dz_to_en` instead of `--csv` if you would rather pull
from the Hub — that repo is private, so `HF_TOKEN` must be set.

---

## Step 5 — Baseline (the gate)

**Do this before training.** NLLB-200 already covers `dzo_Tibt`, so Meta shipped
a Dzongkha translator before you started. That, not zero, is the number to beat.

```bash
python evaluate.py --model facebook/nllb-200-distilled-600M \
    --data /workspace/data/dz_en_bidi --split test \
    --out baseline_own.json

python evaluate.py --model facebook/nllb-200-distilled-600M \
    --flores --out baseline_flores.json
```

Write both numbers down. FLORES-200 is the standard benchmark for this pair, so
that score is what makes your results comparable to published work — and it is
the one you quote to anyone else.

---

## Step 6 — First training run

```bash
tmux new -s train        # so an SSH drop does not kill 5 hours of work

python train.py \
  --data /workspace/data/dz_en_bidi \
  --out /workspace/ckpt \
  --hub-id kinleyrabgay/lotsawa-600m-dz-en \
  --batch 32 --epochs 3 --lr 3e-5 --max-length 64 \
  --eval-steps 4000

# detach: ctrl-b then d       reattach: tmux attach -t train
```

Watch it for the first ten minutes:

```bash
watch -n2 nvidia-smi     # want >90% utilization; if low, raise --batch
```

`train/loss` should fall below ~2.0 within the first few thousand steps. If it
goes NaN you are somehow in fp16 — drop the flag and use bf16.

Interrupted (spot preemption, crash, your laptop closing)? Same command plus
`--resume`.

Expect roughly 3–5 hours for 3 epochs at batch 32 on a 4090. `--eval-steps 4000`
rather than the default 2000 keeps generation-based evaluation from eating an
extra hour.

### The gate

```bash
python evaluate.py --model /workspace/ckpt \
    --data /workspace/data/dz_en_bidi --split test --out ft_own.json
python evaluate.py --model /workspace/ckpt --flores --out ft_flores.json
```

**The fine-tune must clearly beat baseline NLLB on chrF++ in both directions.**
If it does not, stop. More data and bigger models will not rescue a broken setup,
and debugging costs $10 here versus $200 later. Check, in order: that the
language codes are `dzo_Tibt`/`eng_Latn`, that labels are padded with `-100`,
that `forced_bos_token_id` is set, and that the loss actually fell.

---

## Step 7 — Back-translation

Only after the gate passes, because this step uses your dz→en model to generate
the English side. A bad model produces useless synthetic data.

```bash
# smoke test first — read the English it produces
python backtranslate.py --model /workspace/ckpt --mono data/dz_mono.txt \
    --limit 500 --out /tmp/bt_test

# then the full 60,750 sentences (~1 hour)
python backtranslate.py --model /workspace/ckpt --mono data/dz_mono.txt \
    --out /workspace/data/dz_en_bt
```

Read the smoke-test output before committing the hour. If the English is
incoherent, the problem is the model from Step 6, not this script.

---

## Step 8 — Retrain with the synthetic data

```bash
tmux new -s train2

python train.py \
  --data /workspace/data/dz_en_bidi \
  --extra-data /workspace/data/dz_en_bt \
  --out /workspace/ckpt-bt \
  --hub-id kinleyrabgay/lotsawa-600m-dz-en-bt \
  --batch 32 --epochs 3 --lr 3e-5 --max-length 128 \
  --eval-steps 4000
```

`--max-length 128`, not 64. The monolingual sentences run 48 syllables at the
median against the corpus's 9, and the shorter budget would truncate most of
them. This run is slower for the same reason — budget 8–12 hours.

Then score it against Step 6, the same way:

```bash
python evaluate.py --model /workspace/ckpt-bt --flores --out bt_flores.json
```

Expect **en→dz to improve more than dz→en** — the synthetic pairs only add rows
in that direction (+28%). If dz→en regresses, downsample the back-translated
data rather than adding forward-translated English.

---

## Step 9 — Try it by hand

```bash
python translate.py --model /workspace/ckpt-bt --direction dz2en "ང་ཡིག་ཚང་ནང་ མེད།"
python translate.py --model /workspace/ckpt-bt --direction en2dz "I am not in office."
python translate.py --model /workspace/ckpt-bt --direction en2dz \
    "Submit the form by 15 January 2027 and pay Nu. 4,500."
```

That third one is the real test. Your parallel corpus contains **zero** digits,
so before back-translation the model has never seen a number. Check that the
date and the amount survive.

---

## Step 10 — Before you terminate the pod

```bash
huggingface-cli whoami                      # confirm you are logged in
ls -la /workspace/ckpt-bt                   # model files present
cat *_flores.json                           # your results, all in one place
```

`--hub-id` pushes at every save, but **verify the repo on huggingface.co before
terminating.** The network volume outlives the pod only while you keep paying for
it; the Hub repo is the durable copy.

Then **terminate** the pod — do not merely stop it, which bills ~$0.028/hr
indefinitely. Delete the network volume too once you no longer need the
checkpoints (~$10/month for 100GB).

Save the JSON result files off the pod too — they are the record of what you
measured, and re-deriving them costs GPU time.

---

## Per-GPU flags

| GPU | Flags |
|---|---|
| **RTX 4090 (24GB)** | `--batch 32` — drop to `--batch 16 --grad-accum 2` if OOM at `--max-length 128` |
| L40S / A100 / RTX PRO 6000 (48GB+) | `--batch 32`, or higher — you have the room |
| T4 / V100 (no bf16) | `--batch 8 --grad-accum 4 --fp16` |
| NLLB-1.3B (needs 48GB+) | `--model facebook/nllb-200-1.3B --batch 8 --grad-accum 4 --grad-checkpointing --lr 1e-5` |

`--grad-accum` trades speed for memory at a fixed effective batch size: `--batch
16 --grad-accum 2` and `--batch 32 --grad-accum 1` train identically, the former
just slower. Reach for it before lowering the effective batch.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Loss goes NaN | fp16 on NLLB. Use bf16 (the default on Ampere+). |
| `CUDA out of memory` | Lower `--batch`, raise `--grad-accum`, add `--grad-checkpointing`. |
| GPU utilization under 50% | Raise `--batch`; check `--dataloader-num-workers`. |
| Disk full mid-run | `HF_HOME` not set — the model cache went to the container disk. |
| BLEU near 0, chrF++ reasonable | Normal for en→dz. BLEU tokenization is unreliable on Tibetan script; trust chrF++. |
| Both metrics near 0 | `forced_bos_token_id` is wrong — the model is decoding into another language. |
| Training dies on SSH disconnect | You forgot tmux. |
| `DatasetNotFoundError` | Private Hub repo without `HF_TOKEN`. Use `--csv ../dataset.csv`. |

---

## Reference — spacing and orthography

Real published Dzongkha spaces at **phrase** boundaries; this corpus spaces at
**word** boundaries:

| | spaces per syllable |
|---|---|
| `dataset.csv`, as shipped | 0.750 |
| Kuensel (`data/dz_mono.txt`) | 0.212 |
| corpus after respacing | 0.333 |

`respace.py` learned 125 boundary markers — case markers and postpositions like
ལུ, ལས, གིས, ནང, དང — from real Kuensel text, recovering **91.2%** of genuine phrase
boundaries on held-out data. `normalize.py` loads `boundary_markers.txt`
automatically. To re-derive them from a larger monolingual set:

```bash
python respace.py learn --mono data/dz_mono.txt --out boundary_markers.txt
python respace.py validate --mono data/dz_mono.txt --markers boundary_markers.txt
```

The corpus was also shipped with its orthography stripped — 0 of 232,489
Dzongkha sentences contained a shad, and 0 English sentences a capital letter.
`prepare_data.py` restores both by default; `--raw` skips it, for an A/B only.

## Reference — regenerating the monolingual data

`data/dz_mono.txt` (60,750 sentences, 47.3% carrying a numeral) is already built.
To rebuild or widen it:

```bash
python fetch_monolingual.py --out data/dz_mono.txt
```

Its purity rests on the `DZONGKHA_DOMAINS` allowlist in that script, not on the
source dataset's language label — only 1 of 19,227 documents was rejected on
language score versus 12,958 on domain, because Dzongkha and Tibetan share a
script and defeat language ID. Have a native reader spot-check a sample, and
extend the allowlist rather than loosening thresholds.

Attribution: `HuggingFaceFW/finetranslations` (ODC-BY). Source articles remain
their publishers' property — confirm reuse terms before shipping commercially.
