# ELMU2056 Signals and Systems — Option 3: Audio Signal Processing

## Setup

```bash
pip install -r requirements.txt
```

For LaTeX report compilation (optional):
```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-publishers
```

## Run

```bash
python3 main.py
```

All outputs are written to `outputs/plots/` (8 PNG figures) and `outputs/audio/` (6 WAV files).

## Compile Report

```bash
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

## Project Structure

```
m-alp/
├── requirements.txt
├── README.md
├── main.py
├── outputs/
│   ├── plots/      # 8 generated figures
│   └── audio/      # 6 generated WAV files
└── report/
    ├── report.tex
    └── references.bib
```
