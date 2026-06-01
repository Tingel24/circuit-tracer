# Reproducibility Study: BabyLM as Second Language Learners

This repository contains the code and experimental results for the reproducibility study of Edman et al. (2024), *"Are BabyLMs Second Language Learners?"*. The original study proposed that BabyLMs (language models trained on limited, developmentally plausible corpora) can be framed as second-language learners, benefiting from explicit linguistic resources such as paraphrase, grammar, and lexical data.

Our focus was on reproducing the **Half/Half model** (paraphrase data + freeform corpus, 50/50 mix), which was reported as the strongest-performing configuration in the original paper.

> ⚠️ **Note:** This repository does **not** include pretrained models, fine-tuned checkpoints, or dataset preprocessing pipelines beyond those supplied by the original authors.

---

## Repository Structure

* `./` – Code for model training and evaluation
* `./evaluation-pipeline-2024/results/` – Evaluation results for BLiMP, BLiMP-Supplement, partial GLUE, and EWoK
* `./bb24_min(supplied_by_lukas_edman).zip` – Original minimal codebase supplied by the authors

---

## Usage

1. Install dependencies (see `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

2. Train a Half/Half model:

   ```bash
   python train.py --spm_model tokenizer.model --dataset halfhalf.train,halfhalf.dev
   ```

3. Run evaluation:

   ```bash
   python src/evaluate.py --model <path-to-checkpoint> --tasks blimp blimp_supp ewok
   ```

### Results

Find results in `/evaluation-pipeline-2024/results`

Find training and evaluation statistics as graphs [here](https://api.wandb.ai/links/lucafrancis-georg-august-univerit-t-g-ttingen/kh0yus9l)

We show results for 2 models:
- models/3597858/checkpoint-89600 -> Batch Size 64
- models/3597630/checkpoint-22400 -> Batch Size 256

### === BLIMP ===
- checkpoint-22400:
  - blimp_supplement               0.5981
  - blimp_filtered                 0.5900
- checkpoint-89600:
  - blimp_supplement               0.6261
  - blimp_filtered                 0.5975

### === EWOK ===
- checkpoint-22400:
  - ewok_filtered                  0.6884
- checkpoint-89600:
  - ewok_filtered                  0.6641

### === FINETUNE ===
- checkpoint-22400:
  -  cola                           0.3221
  -  boolq                          0.6508
  -  mnli-mm                        0.7679
  -  multirc                        0.6477
  -  mrpc                           0.9135
  -  mnli                           0.7577
  -  GLUE macroaverage              0.6766
- checkpoint-89600:
  -  cola                           0.3344
    -  boolq                          0.6550
  -  mnli-mm                        0.7659
    -  multirc                        0.6328
  -  mrpc                           0.8980
    -  mnli                           0.7637
    -  GLUE macroaverage           0.6750
## Limitations

* Reproduction diverged from original results.
* GLUE evaluation was incomplete due to compute constraints.
* EWoK evaluation pipeline contained a known bug, which inflated some scores.
