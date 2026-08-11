# Community-Based Sarcopenia Screening Tool for Older Adults

This repository contains the source code and locked model artifacts for the web-based screening application accompanying the manuscript.

The application is implemented in Streamlit and provides preliminary sarcopenia risk screening for adults aged 65 years and older. It is intended for research and preliminary screening only. It is not a validated clinical decision-support system, does not provide a diagnosis, and must not be used as the sole basis for treatment or referral decisions.

## Repository contents

- `app.py`: Streamlit application and input-validation logic.
- `best_sarcopenia_lr_model2.pkl`: locked logistic-regression pipeline.
- `best_features_model2.pkl`: ordered model feature list.
- `best_threshold_model2.pkl`: locked classification threshold.
- `model_metadata.pkl`: model metadata used by the application.
- `requirements.txt`: Python dependencies.
- `render.yaml`: optional Render deployment configuration.

## Requirements

- Python 3.11 or a compatible Python version.
- The package versions specified in `requirements.txt`.

## Local installation

```bash
python -m venv .venv
```

Activate the virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Open the local address displayed by Streamlit, usually `http://localhost:8501`.

## Input data

The application accepts CSV and Excel files. Each row represents one participant. The required columns are:

```text
ID, sex, age, EQ5D, pa_aerobic, is_allownc, Chew_Diff,
Prot_Deficiency, N_EN, HE_wc, obe_4class
```

`ID` and `sex` are retained for result management and are not used as model predictors. Detailed definitions and coding rules are displayed in the application under **How to Use** and **Model Info**.

## Output

The downloadable results include the original input columns and:

- `model_score`: estimated model probability.
- `predicted_class`: thresholded binary model output.
- `screening_result`: `Screen positive` or `Screen negative`.

## Important limitations

- The model is restricted to its intended population and setting.
- A screen-positive result is not a diagnosis.
- Model performance may change across populations, locations, and time periods.
- Local validation, calibration monitoring, privacy safeguards, and clinical governance are required before operational use.

## Code availability

This repository is the manuscript-facing public snapshot of the web application. The files are provided together so that reviewers and readers can inspect and run the application using the locked model artifacts.
