# Spanish → Quechua Neural Machine Translation

A from-scratch implementation of the Transformer architecture (Vaswani et al., 2017) applied to the low-resource machine translation task of translating Spanish into Quechua.

---

## Overview

This notebook walks through every stage of building a sequence-to-sequence Transformer: from implementing the core attention mechanism, to training on a real Spanish–Quechua dataset, to evaluating translations with BLEU, chrF++, and COMET. It also compares three tokenization strategies (WordLevel, BPE, and Unigram) to assess their effect on translation quality.

---

## Architecture

The model follows the original encoder-decoder Transformer design with the following components:

- **Scaled Dot-Product Attention** — implements Equation (1) from Vaswani et al., with support for optional causal and padding masks.
- **Multi-Head Attention** — projects queries, keys, and values into multiple subspaces and aggregates the results.
- **Position-wise Feed-Forward Network** — two linear layers with GELU activation and dropout.
- **Sinusoidal Positional Encoding** — fixed encodings added to token embeddings; lower dimensions capture local patterns, higher dimensions capture global structure.
- **Encoder** — stack of `N` encoder layers, each with self-attention + FFN + Pre-LN residual connections.
- **Decoder** — stack of `N` decoder layers with masked self-attention, cross-attention over encoder output, and FFN.
- **Weight Tying** — the output projection shares weights with the source embedding when vocabularies are the same size (Press & Wolf, 2017).
- **Xavier Uniform Initialization** — applied to all linear and embedding layers.

Default hyperparameters:

| Parameter | Value |
|---|---|
| `d_model` | 256 |
| `num_heads` | 4 |
| `d_ff` | 1024 |
| `num_layers` | 4 |
| `dropout` | 0.2 |
| `max_len` | 128 |
| `vocab_size` | 12,000 (per language) |

---

## Dataset

The notebook uses the [`somosnlp-hackathon-2022/spanish-to-quechua`](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua) dataset from HuggingFace, containing parallel Spanish–Quechua sentence pairs split into train, validation, and test sets.

**Preprocessing steps:**
- Remove duplicate pairs.
- Drop empty sentences.
- Filter out pairs where either side exceeds 127 words (to avoid excessively long sequences).

A back-translation data augmentation pipeline is also implemented (Spanish → English → Spanish via MarianMT) to expand the training set, though it is commented out by default due to the computational cost.

---

## Tokenization

Three tokenization strategies are trained and compared, each with a vocabulary of 12,000 tokens:

| Strategy | Description |
|---|---|
| **WordLevel** | Simple whitespace-split vocabulary. Fast but high OOV rate. |
| **BPE** (Byte-Pair Encoding) | Iteratively merges the most frequent byte pairs. Good balance of coverage and compactness. |
| **Unigram** | Probabilistic subword model that selects the most likely segmentation. |

All tokenizers use special tokens `[start]`, `[end]`, `[pad]`, and `[unk]`, and are saved to JSON for reuse.

---

## Training

**Optimizer:** Adam with `β₁=0.9`, `β₂=0.98`, `ε=1e-9`, and weight decay `1e-5`.

**Learning rate schedule:** Linear warmup for the first 1,000 steps, followed by cosine annealing down to `1e-5`.

**Loss:** Cross-entropy with label smoothing (`0.15`) and padding token ignored.

**Gradient clipping:** Max norm of `1.0`.

**Epochs:** 400, with the best checkpoint (lowest validation loss) saved automatically to `Best_model.pth`.

Training and validation losses are logged to CSV files (e.g., `BPE.csv`, `Unigram.csv`) for later analysis.

---

## Decoding

Two decoding strategies are available:

- **Greedy decoding** (`Traducir`) — selects the highest-probability token at each step.
- **Beam search** (`beam_decode`) — maintains a beam of the top-k candidate sequences. The optimized version batches all active beams into a single forward pass per timestep for efficiency.

---

## Evaluation

Translations are evaluated using three metrics:

- **BLEU** (via `sacrebleu`) — n-gram precision with brevity penalty.
- **chrF++** — character n-gram F-score with word order component.
- **COMET** (`Unbabel/wmt22-comet-da`) — neural metric that correlates strongly with human judgments.

Results are saved to a CSV file (`translation_results_2.csv`) that accumulates across runs for easy comparison between experiments.

**BLEU score reference guide:**

| Score | Quality |
|---|---|
| < 10 | Almost useless |
| 10–19 | Hard to convey the gist |
| 20–29 | Understandable, significant errors |
| 30–40 | Intelligible, some mistakes |
| 40–50 | High quality |
| 50–60 | Very high quality, close to human |
| > 60 | Superior to human |

---

## Installation

```bash
pip install seaborn matplotlib transformers tokenizers sacrebleu datasets SentencePiece unbabel-comet torch
```

A CUDA-capable GPU is strongly recommended for training.

---

## File Structure

```
.
├── Codigo.ipynb               # Main notebook
├── BPE/
│   ├── es_tokenizer.json
│   ├── que_tokenizer.json
│   ├── Best_model.pth
│   └── BPE.csv                # Training history
├── Unigram/
│   ├── es_tokenizer.json
│   ├── que_tokenizer.json
│   ├── Best_model.pth
│   └── Unigram.csv
├── Wordlevel/
│   ├── es_tokenizer.json
│   ├── que_tokenizer.json
│   └── wordlevel.csv
├── checkpoint.pt              # Latest checkpoint (for resuming)
├── translation_results_2.csv  # Evaluation scores across runs
└── loss_comparison.pdf        # Combined loss plot
```

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Press, O. & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT. (`sacrebleu`)
- Rei, R. et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task.* WMT.
