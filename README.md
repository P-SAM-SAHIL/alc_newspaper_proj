# alc_newspaper_proj

## English A La Carte (ALC) Embedding Pipeline

English A La Carte (ALC) embedding pipeline for yearly historical newspaper datasets using fastText embeddings and localized bias analysis. 

---

# Features

* Uses pretrained fastText English vectors (`cc.en.300.bin`)
* Generates custom yearly ALC transformation matrices
* Supports:

  * State-level analysis
  * County-level analysis
  * City-level analysis
* Produces:

  * Localized embeddings
  * Bias score tables
  * Reusable `.npz` ALC bundles
* Optimized for large newspaper corpora
* Works well on:

  * Local systems
  * Kaggle notebooks (recommended for large datasets due to 30 GB RAM)

---

# Project Structure

```bash
project/
│
├── alc_year_pipeline.py
├── cleaned_1963.csv
├── dictionaries.json          # optional custom dictionaries
├── outputs/
│
└── cc.en.300.bin              # fastText embeddings
```

---

# System Requirements

## Recommended

| Environment      | RAM   | Notes                 |
| ---------------- | ----- | --------------------- |
| Local PC         | 16GB+ | Small/medium datasets |
| Kaggle           | 30GB  | optional         |
| Google Colab Pro | 25GB+ | Optional              |

---

# Install Dependencies

## Create Environment (Recommended)

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Required Packages

```bash
pip install numpy tqdm nltk gensim
```

---

# Download fastText Embeddings

The pipeline uses Facebook AI fastText English vectors.

## Kaggle / Linux / Colab

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

## Windows PowerShell

Install `wget` or download manually from:

```text
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
```

Then extract:

```powershell
gzip -d cc.en.300.bin.gz
```

You should finally have:

```bash
cc.en.300.bin
```

---

# Input CSV Format

Expected columns:

```text
article_id,date,newspaper_name,headline,article,LCCN,State,County,City
```

Example:

```csv
1,1963-01-01,Example News,Headline text,Article text...,12345,Texas,Dallas,Dallas
```

---

# Run on Local System

## Basic Run

```bash
python alc_year_pipeline.py \
  --csv cleaned_1963.csv \
  --year 1963 \
  --fasttext cc.en.300.bin \
  --out-dir outputs
```

---

## Recommended Run

```bash
python alc_year_pipeline.py \
  --csv cleaned_1963.csv \
  --year 1963 \
  --fasttext cc.en.300.bin \
  --dict-json dictionaries.json \
  --out-dir outputs \
  --window-size 5 \
  --min-count 20
```

---

# Run on Kaggle (Recommended)

Kaggle provides approximately **30 GB RAM**, which is highly useful for:

* loading fastText vectors
* large corpus processing
* generating ALC matrices

---

# Kaggle Setup

## Step 1 — Upload Files

Upload:

* `alc_year_pipeline.py`
* `cleaned_1963.csv`
* optional `dictionaries.json`

---

## Step 2 — Install Dependencies

```python
!pip install gensim nltk tqdm
```

---

## Step 3 — Download fastText

```python
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz

# unzip
!gunzip cc.en.300.bin.gz
```

---

## Step 4 — Run Pipeline

```python
!python alc_year_pipeline.py \
  --csv cleaned_1963.csv \
  --year 1963 \
  --fasttext cc.en.300.bin \
  --dict-json dictionaries.json \
  --out-dir outputs \
  --window-size 5 \
  --min-count 20
```

---

# Reusing Existing ALC Weights

If you already generated yearly weights:

```bash
python alc_year_pipeline.py \
  --csv cleaned_1963.csv \
  --year 1963 \
  --fasttext cc.en.300.bin \
  --alc-weights outputs/global_alc_vectors_1963.npz \
  --out-dir outputs
```

This avoids retraining the transformation matrix.

---

# Output Files

The pipeline generates:

| File                             | Description              |
| -------------------------------- | ------------------------ |
| `A_1963_news.npy`                | Transformation matrix    |
| `global_alc_vectors_1963.npz`    | Reusable ALC bundle      |
| `state_year_bias_table_1963.csv` | Final bias analysis      |
| `dictionaries_used.json`         | Cleaned dictionary terms |

---

# Final Output Table

Generated file:

```text
state_year_bias_table_1963.csv
```

Columns:

| Column            | Description                           |
| ----------------- | ------------------------------------- |
| State             | Geographic region                     |
| Year              | Dataset year                          |
| Diff bias score   | Difference between group similarities |
| Bias score_group1 | Similarity score for group 1          |
| Bias score_group2 | Similarity score for group 2          |
| Bias concept      | Bias comparison category              |

---

# Example Bias Concepts

* Black-negative / White-negative
* Black-positive / White-positive
* Men-positive / Women-positive
* Rich-negative / Poor-negative

---

# Performance Tips

## Faster Processing

Reduce:

```bash
--max-regression-words
--max-global-alc-words
```

Example:

```bash
--max-regression-words 50000
```

---

## Lower Memory Usage

Use:

```bash
--fasttext-limit 200000
```

Only works for `.vec` or `.txt` models.

---

# Enable OCR Fuzzy Matching

Useful for noisy historical OCR datasets.

```bash
--enable-fuzzy
```

Example:

```bash
python alc_year_pipeline.py \
  --csv cleaned_1963.csv \
  --year 1963 \
  --fasttext cc.en.300.bin \
  --enable-fuzzy
```

---

# Full Advanced Example

```bash
python alc_year_pipeline.py \
  --csv cleaned_1963.csv \
  --year 1963 \
  --fasttext cc.en.300.bin \
  --dict-json dictionaries.json \
  --out-dir outputs \
  --window-size 5 \
  --min-count 20 \
  --max-regression-words 100000 \
  --max-global-alc-words 200000 \
  --enable-fuzzy
```

---

# Notes

* `.bin` fastText models are recommended because they support subword OOV handling.
* Historical newspaper OCR quality can affect embedding quality.


---

# Citation

If you use this pipeline in research, cite:

* fastText by Facebook AI Research
* A La Carte Embedding methodology

---

# License

MIT License (recommended)
