# Evaluation

This README explains how to run evaluation scripts in the `flowise` directory of your Phoenix workspace.

## Prerequisites

- Python 3.10+ (recommended)
- All required packages installed (see below)
- Phoenix backend and frontend running (optional for full integration)

## Setup

1. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**

   If you have a `requirements.txt` file in `flowise`, run:

   ```bash
   pip install -r requirements.txt
   ```

   If not, install any required packages manually (e.g., pandas, numpy, etc.).

## Running Phoenix

```
python -m phoenix.server.main serve
```

## Running Evaluations

To run an evaluation script, use the following command:

```bash
cd flowise
python experiment1.py
```

## Example Dataset

Download the dataset here:

[Download CSV](data/evaluation_dataset_1.csv)


## Typical Workflow

1. Connect your Flowise instance to Phoenix.

2. Go to **Datasets and Experiments** in Phoenix and create a **new dataset**. Upload your evaluation dataset.

3. Navigate to the script directory and update the dataset name:Open `experiment1.py` and replace `SOURCE_DATASET= **dataset name created in Phoenix**`.

4. Run the script using Python.

5. After execution, go to **Phoenix → Experiments** to view the **evaluation results**.


## Troubleshooting

- If you see missing package errors, install them with `pip install <package>`.
- If you want to run in a virtual environment, activate it before running scripts.
- For integration with Phoenix UI, ensure the backend and frontend are running as described in the main project README.
