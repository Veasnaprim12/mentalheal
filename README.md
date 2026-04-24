# Mental Health Statement Classification

This project builds and compares multiple NLP approaches for classifying mental health statements into status categories.

## Project Goal

The workflow in this repository focuses on:

- Cleaning and preprocessing text statements
- Exploring class distribution and dataset quality
- Training classical ML models (Logistic Regression and XGBoost)
- Experimenting with transformer-based deep learning models (BERT and DistilBERT)
- Saving trained artifacts for reuse

## Repository Structure

- `mental_health.csv`: Original dataset
- `mental_health.ipynb`: Data cleaning, EDA, and text preprocessing
- `cleaned_statements.csv`: Cleaned text output from preprocessing notebook
- `ML_part.ipynb`: TF-IDF feature extraction, Logistic Regression, XGBoost, and evaluation
- `cleaned_statements2.csv`: Processed dataset with encoded labels
- `deep_learnning.ipynb`: Transformer experiments using Hugging Face
- `vectorizer.pkl` and `vectorizer1.pkl`: Saved vectorizers
- `noted/model_logistic.pkl`: Saved Logistic Regression model
- `noted/xgboost_model.pkl`: Saved XGBoost model
- `logs/`: Training logs

## End-to-End Workflow

1. Open and run `mental_health.ipynb` to clean the source data and generate `cleaned_statements.csv`.
2. Run `ML_part.ipynb` to:
   - Encode labels
   - Vectorize text using TF-IDF
   - Train and evaluate Logistic Regression and XGBoost models
   - Export model artifacts and `cleaned_statements2.csv`
3. Run `deep_learnning.ipynb` for transformer-based experimentation (BERT, DistilBERT).

## Requirements

Recommended Python version:

- Python 3.10 or later

Suggested packages:

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn
- xgboost
- transformers
- datasets
- torch
- joblib
- jupyter
- ipykernel

Install dependencies:

```bash
pip install -r requirements.txt
```

## NLTK Setup

The preprocessing notebook downloads required NLTK resources. If needed, run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Model Outputs

Saved artifacts include:

- TF-IDF vectorizer (`vectorizer.pkl`)
- Logistic Regression model (`noted/model_logistic.pkl`)
- XGBoost model (`noted/xgboost_model.pkl`)

## Reproducibility Notes

- Keep dataset files in the project root so notebook paths resolve correctly.
- Maintain consistent random states when comparing model runs.
- Use stratified train/test splitting for fair class distribution.

## Preparing a GitHub Pull Request

Before opening a PR:

1. Ensure notebooks run in order without missing dependencies.
2. Confirm generated files are intentional and not redundant.
3. Review large binary files (`.pkl`, logs) and decide whether they should be versioned.
4. Fill out `.github/PULL_REQUEST_TEMPLATE.md` with clear details.
5. Add a short PR summary covering:
   - What was changed
   - Why it was changed
   - How it was validated

Suggested commit message examples:

- `docs: add project README with workflow and setup`
- `chore: document notebook pipeline and model artifacts`

## Next Improvements

- Add notebook-to-script training pipelines for automation.
- Add a lightweight inference script to classify new statements.
- Track metrics across runs in a single experiment report.
