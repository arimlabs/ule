# Entropy of Ukrainian — Experiment Server

Web application and analysis code for the Shannon's Guessing Game replication study for Ukrainian, as described in the paper *Entropy of Ukrainian* (ARIMLABS.AI).

> **Note on AI assistance:** Large language models of the Claude family were used extensively during development for code generation purposes.

---

## Repository structure

```
app/                        FastAPI web application (experiment server)
dataset_export/             Anonymized dataset used in the paper
notebooks/
  dataset_collection.ipynb  Scrapes and preprocesses source articles from Ukrainska Pravda
  llm_benchmarking.ipynb    LLM BPC evaluation and dataset diversity analysis
scripts/
  entropy_analysis.py       Main entropy analysis and bootstrap
  ingest_dataset.py         Populates the database with sentences
  export_dataset.py         Exports anonymized dataset CSVs
conf.py                     Experiment configuration constants
main.py                     FastAPI entry point
```

---

## Datasets

- [Source articles](https://huggingface.co/datasets/a-l-o/shortnews) — 5 Ukrainska Pravda articles used as sentence source
- [Experiment results](https://huggingface.co/datasets/a-l-o/ule_results) — anonymized sentences, sessions, and guesses

---

## Running the experiment server

### Prerequisites

- Python 3.12+
- PostgreSQL instance
- [Upstash Redis](https://upstash.com/) account (used for participant ID validation)
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Setup

1. **Install dependencies**

   ```bash
   uv sync
   # or: pip install -r requirements.txt
   ```

2. **Configure environment** — create a `.env` file:

   ```
   POSTGRESQL_URL=postgresql+asyncpg://user:password@host:5432/dbname
   UPSTASH_REDIS_URL=https://...
   UPSTASH_REDIS_TOKEN=...
   ```

3. **Ingest the sentence dataset**

   ```bash
   python3 scripts/ingest_dataset.py
   ```

   This pulls sentences from the HuggingFace dataset, preprocesses them and inserts them into the database.

4. **Start the server**

   ```bash
   fastapi dev main.py         # development
   fastapi run main.py         # production
   ```

   The experiment UI is served at `/`.

---

## Running the analysis

Requires `dataset_export/` to be populated (either from the published dataset or via `export_dataset.py`).

```bash
python3 scripts/entropy_analysis.py
```

This reproduces all results and figures from the paper:
- Entropy bounds by position
- Bootstrap confidence intervals (session-level and sentence-level)
- Binomial cheater detection + trim-sensitivity analysis
- Character frequency comparison

Output figures are written to `scripts/graphs/`.

---

## Exporting the anonymized dataset

```bash
EXPORT_SALT=<your-secret-salt> python3 scripts/export_dataset.py
```

To generate a salt:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Produces three CSVs in `scripts/export/`:

| File | Contents |
|---|---|
| `sentences.csv` | Sentence text, type, length |
| `sessions.csv` | Sessions with anonymized participant IDs |
| `guesses.csv` | All character guesses (positions ≥ 70) |

Participant IDs are replaced with `sha256(salt + original_id)[:8]` integers.