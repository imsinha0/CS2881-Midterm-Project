# CS2881 Midterm Project

## Installation and Setup

1. **Install [uv package manager](https://docs.astral.sh/uv/getting-started/installation/):**

   Follow the instructions at the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Sync dependencies:**

   ```bash
   uv sync
   ```

3. **Build the handbook data index:**

   ```bash
   uv run python data/data_extractor.py
   ```

## Reproducing Main Results


4. **Download NLTK data ("punkt_tab"):**

   ```bash
   uv run python
   ```

   Then, in the Python prompt:

   ```python
   import nltk
   nltk.download('punkt_tab')
   ```

   Press <kbd>Ctrl</kbd>+<kbd>D</kbd> (or exit) to leave the Python prompt.

5. **Run the main experiment:**

   ```bash
   uv run python main.py
   ```

## Reproducing Extension Results

To reproduce the extension experiments, run:

```bash
uv run python extensions.py
```

## Results Analysis

To view an analysis of the results, visit the following Overleaf document:

[Project Analysis on Overleaf](https://www.overleaf.com/read/qjxtvwdvmrsk#cded7a)

---

## Acknowledgements

We did this work as part of the class ["Machine Learning Theory Seminar"](https://boazbk.github.io/mltheoryseminar/).