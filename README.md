# Rare-word-Retention-in-Abstractive-Summarization-via-Copy-aware-Transformer
Author: **Sara Hodaei**  
Rare-Word Retention in Abstractive Summarisation via Hybrid Copy-Aware Transformer. This project explores the integration of pointer–generator and coverage mechanisms into BART-base, combined with span-aware decoding and entity-focused evaluation, to improve factuality and rare-word/entity retention in CNN/DailyMail summarisation.

## Project Overview

This repository contains the code, analysis, and dissertation for the MSc Data Science project:

The project investigates two persistent challenges in abstractive summarisation:

- **Hallucination**: generation of unsupported or fabricated facts.  
- **Rare-word omission**: exclusion or misrepresentation of critical entities such as names, numbers, and locations.  

### Key Contributions

1. Augmented **BART-base** with a pointer–generator mechanism and coverage loss.  
2. Introduced **span-aware decoding** (inspired by CopyNext and SeqCopyNet) at inference.  
3. Applied **entity-focused evaluation** (precision, recall, F1, UCER) in addition to ROUGE.  

Results on the CNN/DailyMail dataset show consistent ROUGE gains and up to **80% reduction in unsupported content entity rate (UCER)** compared with the baseline.

---

Repository Structure

```text
copy-aware-summarization-thesis/
│
├── README.md                → Project overview
├── LICENSE                  → MIT License
├── requirements.txt          → Python dependencies
│
├── thesis/
│   └── SaraHodaei_Thesis.pdf
│
├── notebooks/
│   └── CopyAware.ipynb
│
├── src/
│   └── copyaware.py
│
├── data/
│   └── README.md            → How to obtain CNN/DailyMail dataset
│
└── docs/
    └── index.md             → GitHub Pages landing page


Environment

Developed and tested on Google Colab Pro with:

Python 3.10

PyTorch 2.6.0 + CUDA 12.4

Transformers 4.44.2

Datasets 3.0.1

A full list of dependencies (with pinned versions) is provided in requirements.txt
.

Getting Started

Clone the repository and install dependencies:

git clone https://github.com/YOUR_USERNAME/copy-aware-summarization-thesis.git
cd copy-aware-summarization-thesis
pip install -r requirements.txt


Download the CNN/DailyMail dataset:

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

Usage

Open the Colab notebook for experiments:
notebooks/CopyAware.ipynb

Read the full dissertation:
thesis/SaraHodaei_Thesis.pdf


License

This project is released under the MIT License
.

