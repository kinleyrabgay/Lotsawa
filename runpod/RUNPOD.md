# Lotsawa — RunPod training runbook

Full sequence, in order. Roughly 5–13 GPU hours and $4–12 end to end on an
RTX 4090.

---

## Step 0 — Decide this before you spend money

`normalize.py` sets `SHAD_AFTER_GA = True`, writing a shad after syllables
ending in ག/ཀ (`འདུག།`, `དོ་ག།`). Traditional Tibetan orthography omits it there;
modern Dzongkha practice varies. This affects **~16%** of the corpus, because
ནུག, འདུག and the question particle ག are among the most common final syllables.

Confirm against the Dzongkha Development Commission style guide or a native
reader, then set the constant. Changing it later means retraining from scratch.

Check the normalizer's output while you are at it:

```bash
python normalize.py ../dataset.csv | head -60
```

---

## Step 1 — Create the pod

| Setting | Value |
|---|---|
| GPU | **RTX 5090 32GB** — $0.99/hr (or RTX 4090 24GB, $0.74/hr) |
| Template | CUDA **12.8+** / PyTorch **2.7+** for the 5090 — see below |
| Network volume | **50GB**, mounted at `/workspace` |
| Container disk | 20GB is fine — nothing large is written to it |

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

> **Blackwell needs a newer template.** RTX 5090, RTX PRO 4000/4500/6000 and
> PRO 6000 MIG are Blackwell (sm_120). A CUDA 12.4 template has no kernels for
> them and fails with `no kernel image is available for execution on the
> device`. Choose a template whose tag shows **CUDA 12.8+ and PyTorch 2.7+**,
> and run the verification in Step 3 before starting a long job. The RTX 4090
> (Ada) and 3090 (Ampere) work on CUDA 12.1+ and carry no such risk.

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
huggingface-cli login --token $HF_TOKEN
nvidia-smi                            # confirm the GPU is what you paid for
```

`requirements.txt` omits `torch` on purpose: the template ships a build compiled
against the pod's exact CUDA version, and letting pip replace it with a wheel for
a different CUDA breaks at the first kernel launch.

**Verify the GPU actually runs a kernel before starting a multi-hour job.**
`torch.cuda.is_available()` can return `True` on a card whose architecture the
build does not support, so run a real matmul:

```bash
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('gpu  ', torch.cuda.get_device_name(0))
print('sm   ', torch.cuda.get_device_capability(0))
print('bf16 ', torch.cuda.is_bf16_supported())
x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
print('matmul ok:', float((x @ x).sum()) == float((x @ x).sum()))
"
```

On a 5090 expect `sm (12, 0)` and `cuda 12.8` or higher. If the matmul raises
`no kernel image is available`, the template is too old — destroy the pod and
pick a CUDA 12.8+ one.

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
