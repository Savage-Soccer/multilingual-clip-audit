# Beyond Performance Gaps: A Geometric Audit of Multilingual CLIP


## The Problem

CLIP achieves **68.6% R@1** for English image retrieval.  
For Hindi it achieves **0.4%** — worse than random.  
This project traces the exact causal chain, down to the tokenizer.

## Key Numbers

| Language | R@1 (CLIP) | CKA ↑ | Procrustes ↓ | Hubness ↓ |
|----------|-----------|-------|--------------|-----------|
| English  | 68.6%     | 1.000 | 0.000        | 2.1       |
| Arabic   | 3.2%      | 0.284 | 0.741        | 87.3      |
| Hindi    | **0.4%**  | **0.112** | **0.886** | **312.8** |

- Hindi BPE tokenization: **23 fragments**, 8 undecodable bytes (English: 6 clean tokens)
- Hubness: **492 of 500** queries return the same image
- GeoAlign post-hoc projection: **29% median rank improvement** from 100 pairs, no retraining

## Notebooks

| # | Notebook | What it does |
|---|----------|-------------|
| 1 | `phase1_2_baseline_retrieval` | Cosine similarity + R@1/5/10/MRR across 5 languages |
| 2 | `phase3_geometric_audit` | CKA, Procrustes disparity, Hubness, UMAP |
| 3 | `phase4_cross_model_ablation` | CLIP vs SigLIP ViT-B/16 vs SigLIP SO400M |
| 4 | `phase5_tokenizer_analysis` | BPE fragment counting, undecodable byte evidence |
| 5 | `phase6_romanization_mitigation` | Devanagari romanization +300% relative R@1 |
| 6 | `part2_geoalign_spectral_alignment` | Linear projection correction, data efficiency analysis |

## Setup
```bash
pip install torch transformers open_clip_torch umap-learn indic-transliteration scipy
```

All notebooks run on **Google Colab free tier**.  
Phases 1–2 need a T4 GPU for encoding. Phases 3–6 are CPU-only.

## Dataset

[XM3600 Crossmodal-3600](https://google.github.io/crossmodal-3600/) —
500 images with professionally annotated native captions in EN, ES, HI, AR, ZH.
