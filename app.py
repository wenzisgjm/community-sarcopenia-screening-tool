from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

ARTIFACTS = {
    "model": APP_DIR / "best_sarcopenia_lr_model2.pkl",
    "features": APP_DIR / "best_features_model2.pkl",
    "threshold": APP_DIR / "best_threshold_model2.pkl",
    "metadata": APP_DIR / "model_metadata.pkl",
}

EXPECTED_FEATURES = [
    "age",
    "EQ5D",
    "pa_aerobic",
    "is_allownc",
    "Chew_Diff",
    "Prot_Deficiency",
    "N_EN",
    "HE_wc",
    "obe_4class",
]

MODEL_VERSION = "1.0"
LAST_UPDATED = "August 2026"
VALIDATION_N = 1156
REPOSITORY_URL = "https://github.com/wenzisgjm/community-sarcopenia-screening-tool"

BINARY_FEATURES = ["pa_aerobic", "is_allownc", "Chew_Diff", "Prot_Deficiency"]
NUMERIC_FEATURES = EXPECTED_FEATURES.copy()

FIELD_DESCRIPTIONS = {
    "ID": "Participant identifier, used to locate screen-positive individuals; not used for model prediction.",
    "sex": "Sex, retained for participant identification and result management; not used for model prediction.",
    "age": "Age (years).",
    "EQ5D": "Health utility index derived from the EQ-5D health questionnaire.",
    "pa_aerobic": (
        "Whether the participant performs at least 150 minutes of moderate-intensity physical activity "
        "per week, at least 75 minutes of vigorous-intensity physical activity per week, or an equivalent "
        "combination of moderate- and vigorous-intensity activity (1 minute of vigorous activity is "
        "equivalent to 2 minutes of moderate activity); yes=0, no=1."
    ),
    "is_allownc": "Whether the participant receives livelihood assistance or subsidies; yes=1, no=0.",
    "Chew_Diff": "Whether the participant has difficulty chewing; yes=1, no=0.",
    "Prot_Deficiency": "Whether protein intake is insufficient; below 60 g/day for men and below 50 g/day for women. Enough=0, Insufficient=1.",
    "N_EN": "Daily energy intake (kcal/day).",
    "HE_wc": "Waist circumference (cm).",
    "obe_4class": (
        "Four-category weight status: Normal weight=0 (18.5 <= BMI < 23 kg/m2); "
        "Underweight=1 (BMI < 18.5 kg/m2); Overweight=2 (23 <= BMI < 25 kg/m2); "
        "Obesity=3 (BMI >= 25 kg/m2)."
    ),
}

INPUT_REQUIREMENTS = {
    "ID": "Participant identifier; required for result management and not used by the model.",
    "sex": "Participant sex; retained for result management and not used by the model.",
    "age": "Numeric age in years; must be 65 years or older.",
    "EQ5D": "Numeric EQ-5D health utility index.",
    "pa_aerobic": "Binary code: meets the specified aerobic activity recommendation=0, does not meet it=1.",
    "is_allownc": "Binary code: receives livelihood assistance or subsidies=1, does not receive them=0.",
    "Chew_Diff": "Binary code: difficulty chewing=1, no difficulty chewing=0.",
    "Prot_Deficiency": "Binary code: insufficient protein intake=1, sufficient protein intake=0.",
    "N_EN": "Numeric daily energy intake in kcal/day.",
    "HE_wc": "Numeric waist circumference in cm.",
    "obe_4class": "Weight-status code: normal weight=0, underweight=1, overweight=2, obesity=3.",
}


st.set_page_config(
    page_title="Community-Based Sarcopenia Screening Tool for Older Adults",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; max-width: 1180px; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 8px;
        padding: 14px 16px;
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    missing = {name: path for name, path in ARTIFACTS.items() if not path.exists()}
    if missing:
        return None, None, None, None, missing, None

    try:
        model = joblib.load(ARTIFACTS["model"])
        features = joblib.load(ARTIFACTS["features"])
        threshold = float(joblib.load(ARTIFACTS["threshold"]))
        metadata = joblib.load(ARTIFACTS["metadata"])
    except Exception as exc:
        return None, None, None, None, {}, exc

    return model, list(features), threshold, metadata, {}, None


def predict(model, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    prob = model.predict_proba(data)[:, 1]
    pred = (prob >= threshold).astype(int)
    out = data.copy()
    out["model_score"] = prob
    out["predicted_class"] = pred
    out["screening_result"] = np.where(pred == 1, "Screen positive", "Screen negative")
    return out


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def validate_batch_data(data: pd.DataFrame, feature_columns: list[str]):
    validated = data.copy()
    errors = []
    warnings = []

    if validated.empty:
        errors.append("The uploaded file contains no participant rows.")
        return validated, errors, warnings

    for column in feature_columns:
        original = validated[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        if invalid_count:
            errors.append(f"{column} contains {invalid_count} non-numeric value(s).")
        validated[column] = converted

    if validated["age"].isna().any():
        errors.append("Age is required for every participant and cannot be missing.")
    elif (validated["age"] < 65).any():
        count = int((validated["age"] < 65).sum())
        errors.append(
            f"{count} participant(s) are younger than 65 years. The model should only be used within its intended population."
        )

    for column in BINARY_FEATURES:
        invalid_codes = sorted(
            validated.loc[validated[column].notna() & ~validated[column].isin([0, 1]), column].unique().tolist()
        )
        if invalid_codes:
            errors.append(f"{column} contains invalid code(s): {invalid_codes}. Allowed codes are 0 and 1.")

    weight_codes = validated.loc[
        validated["obe_4class"].notna() & ~validated["obe_4class"].isin([0, 1, 2, 3]), "obe_4class"
    ].unique().tolist()
    if len(weight_codes):
        errors.append(
            f"obe_4class contains invalid code(s): {sorted(weight_codes)}. Allowed codes are 0, 1, 2, and 3."
        )

    missing_counts = validated[feature_columns].isna().sum()
    imputed_columns = [f"{column} ({int(count)})" for column, count in missing_counts.items() if count and column != "age"]
    if imputed_columns:
        warnings.append(
            "Missing model inputs will be imputed by the locked preprocessing pipeline: " + ", ".join(imputed_columns) + "."
        )

    if validated["ID"].isna().any():
        warnings.append("One or more participant identifiers are missing, which may make follow-up difficult.")
    if validated["ID"].duplicated().any():
        warnings.append("Duplicate participant identifiers were detected. Confirm that each row represents the intended participant record.")

    return validated, errors, warnings


st.title("Community-Based Sarcopenia Screening Tool for Older Adults")
st.caption("Strong bodies build strong communities.")
st.info(
    "Target users: adults aged 65 years and older.\n\n"
    "Use case: preliminary sarcopenia risk screening in community or primary care settings."
)
st.warning(
    "Research and preliminary screening use only. This tool is a research demonstration and preliminary "
    "sarcopenia screening aid. It is not a validated clinical decision-support system, does not provide a "
    "diagnosis, and must not be used as the sole basis for treatment or referral decisions."
)

with st.spinner("Loading the prediction model..."):
    model, best_features, best_threshold, metadata, missing_artifacts, load_error = load_artifacts()

if missing_artifacts:
    st.error("Required model files were not found.")
    st.write("Run the model export script first, then place the generated files in the same folder as app.py.")
    st.dataframe(
        pd.DataFrame(
            [{"File": path.name, "Status": "Missing", "Expected location": str(path)} for path in missing_artifacts.values()]
        ),
        width="stretch",
    )
    st.stop()

if load_error is not None:
    st.error(
        "The model files could not be loaded. Please confirm that the Python version, "
        "scikit-learn version, and model files are compatible."
    )
    st.exception(load_error)
    st.stop()

if best_features != EXPECTED_FEATURES:
    st.warning(
        "The feature order stored in the model files differs from the expected Model_2 order. "
        "Predictions will use the order stored in the model files."
    )

tab_screening, tab_guide, tab_transparency, tab_governance = st.tabs(
    ["Screening", "How to Use", "Model Info", "Governance"]
)

with tab_screening:
    st.subheader("Batch Screening")
    st.info(
        "Upload a CSV or Excel file containing participant identifiers, sex, and the required screening "
        "variables. The tool will identify screen-positive participants who should receive further assessment."
    )
    st.write("Upload a CSV or Excel file containing at least the following columns:")
    required_batch_columns = ["ID", "sex"] + best_features
    st.code(", ".join(required_batch_columns))
    st.caption(
        "Note: ID and sex are retained for participant identification and result review. "
        "They are not used to calculate model risk."
    )

    uploaded = st.file_uploader("Select a file for batch prediction", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            batch_df = read_uploaded_table(uploaded)
        except Exception as exc:
            st.error(f"The uploaded file could not be read: {exc}")
        else:
            missing_cols = [col for col in required_batch_columns if col not in batch_df.columns]

            if missing_cols:
                st.error("The uploaded file is missing required columns: " + ", ".join(missing_cols))
            else:
                validated_df, validation_errors, validation_warnings = validate_batch_data(batch_df, best_features)
                for message in validation_warnings:
                    st.warning(message)
                for message in validation_errors:
                    st.error(message)

                if not validation_errors:
                    batch_input = validated_df[best_features].copy()
                    prediction_details = predict(model, batch_input, best_threshold)
                    prediction_details.insert(0, "sex", validated_df["sex"].values)
                    prediction_details.insert(0, "ID", validated_df["ID"].values)

                    total_n = len(prediction_details)
                    positive_n = int(prediction_details["predicted_class"].sum())
                    negative_n = total_n - positive_n

                    st.warning(
                        f"Screening rule: A model score of {best_threshold:.4%} or higher is classified as screen positive. "
                        "A screen-positive result does not confirm sarcopenia. Assessment using established measures of "
                        "muscle strength, muscle mass, and physical performance is required."
                    )
                    st.caption(
                        "The model score is a screening estimate and should not be interpreted as an individual's confirmed "
                        "clinical probability of sarcopenia."
                    )

                    metric_a, metric_b, metric_c = st.columns(3)
                    metric_a.metric("Total Participants", total_n)
                    metric_b.metric("Screen Positive", positive_n)
                    metric_c.metric("Screen Negative", negative_n)

                    st.bar_chart(prediction_details["screening_result"].value_counts())

                    result_columns = ["ID", "sex"] + best_features + ["model_score", "screening_result"]
                    result_table = prediction_details[result_columns].copy()

                    def highlight_screen_positive(row):
                        if row["screening_result"] == "Screen positive":
                            return ["background-color: #fee2e2; color: #991b1b; font-weight: 600"] * len(row)
                        return [""] * len(row)

                    styled_result = result_table.style.apply(highlight_screen_positive, axis=1).format(
                        {"model_score": "{:.1%}"}
                    )
                    st.dataframe(styled_result, width="stretch", hide_index=True)

                    csv_bytes = result_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        "Download Screening Results (CSV)",
                        data=csv_bytes,
                        file_name="sarcopenia_screening_results.csv",
                        mime="text/csv",
                        width="stretch",
                    )

with tab_guide:
    st.subheader("Intended Users")
    st.write(
        "This tool is intended for trained community nurses, primary care physicians, public health practitioners, "
        "and qualified research staff conducting sarcopenia risk screening among adults aged 65 years and older."
    )

    st.subheader("Recommended Workflow")
    st.markdown(
        """
        1. Collect and verify the required screening variables.
        2. Remove unnecessary direct identifiers and upload a correctly formatted CSV or Excel file.
        3. Review participants classified as screen positive.
        4. Confirm findings using established assessments of muscle strength, muscle mass, and physical performance.
        5. Refer participants for clinical assessment according to local protocols when appropriate.
        """
    )

    st.subheader("Training Before Use")
    st.write(
        "Users should understand the variable definitions and coding rules, the meaning of a screen-positive result, "
        "the limits of model-generated scores, and their local confirmatory assessment and referral procedures. "
        "The tool should only be introduced within an organization after an appropriate clinical or research lead has "
        "reviewed these requirements with users."
    )

    st.subheader("Input Requirements")
    input_table = pd.DataFrame(
        {"Variable": required_batch_columns, "Requirement": [INPUT_REQUIREMENTS[field] for field in required_batch_columns]}
    )
    st.dataframe(input_table, width="stretch", hide_index=True)
    st.caption(
        "Missing model inputs other than age are handled by the locked preprocessing pipeline using development-data "
        "medians for continuous variables and most-frequent values for categorical variables. Extensive missingness may "
        "reduce reliability and should be reviewed before interpretation."
    )

    template_csv = pd.DataFrame(columns=required_batch_columns).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Input Template (CSV)",
        data=template_csv,
        file_name="sarcopenia_screening_input_template.csv",
        mime="text/csv",
    )

    st.subheader("Interpreting Results")
    st.warning(
        "Screen positive means that the model score met or exceeded the screening threshold. It does not confirm "
        "sarcopenia. Screen negative does not rule out sarcopenia when symptoms, functional decline, or clinical concern "
        "are present. Clinical judgment and established assessment procedures remain necessary."
    )

with tab_transparency:
    st.subheader("Model Information")
    info = {
        "Algorithm": metadata.get("final_algorithm", "LogisticRegression"),
        "Model version": MODEL_VERSION,
        "Current status": "Research demonstration and preliminary screening aid",
        "Predictor set": metadata.get("final_feature_set", "Model_2"),
        "Number of predictors": len(best_features),
        "Development sample": f"{metadata.get('development_year', 2022)} dataset, n={metadata.get('development_n', 889)}",
        "Temporal validation sample": f"{metadata.get('validation_year', 2024)} dataset, n={metadata.get('validation_n', VALIDATION_N)}",
        "Screening threshold": f"{best_threshold:.6f} ({best_threshold:.4%})",
        "Threshold selection": metadata.get(
            "threshold_method", "Youden index based on out-of-fold predictions from the development dataset"
        ),
        "scikit-learn version": metadata.get("sklearn_version", "Unknown"),
        "Last updated": LAST_UPDATED,
    }
    info_table = pd.DataFrame(
        [(item, str(details)) for item, details in info.items()],
        columns=["Item", "Details"],
    )
    st.table(info_table)

    st.subheader("Threshold Interpretation")
    st.info(
        "The locked threshold was selected to balance sensitivity and specificity for preliminary screening. It is not "
        "a diagnostic cutoff and may require recalibration before use in populations that differ from the development "
        "and temporal validation samples."
    )

    field_table = pd.DataFrame(
        {
            "Variable": ["ID", "sex"] + best_features,
            "Explanation": [FIELD_DESCRIPTIONS[field] for field in ["ID", "sex"] + best_features],
        }
    )
    st.subheader("Data Field Descriptions")
    st.caption("Important: Numeric codes in the uploaded file must match the codes used in the original model development data.")
    st.dataframe(field_table, width="stretch", hide_index=True)

    st.subheader("Known Limitations")
    st.write(
        "Performance may differ across settings, demographic groups, data-collection methods, and disease prevalence. "
        "The tool has not been established as a standalone clinical decision-support system. Model scores should not be "
        "interpreted as confirmed individual probabilities unless calibration has been demonstrated in the intended local population."
    )

with tab_governance:
    st.subheader("Intended and Prohibited Uses")
    st.write(
        "The intended use is preliminary sarcopenia risk screening and research demonstration in community or primary "
        "care settings for adults aged 65 years and older. The tool must not be used to diagnose sarcopenia, make "
        "autonomous treatment decisions, replace professional judgment, or screen people outside the intended population."
    )

    st.subheader("Data Privacy")
    st.write(
        "The application code does not intentionally write uploaded participant files to persistent application storage. "
        "Files are processed during the active application session. Users should remove unnecessary direct identifiers "
        "and comply with institutional policies, applicable data-protection requirements, and the hosting provider's "
        "terms before uploading any participant data."
    )

    st.subheader("Model Updating and Version Control")
    st.write(
        "The model is not automatically retrained or updated. Any future revision requires documented validation, a new "
        "version number, updated performance and limitation statements, and review before deployment. The model version "
        "and update date shown on this page identify the currently deployed release."
    )

    st.subheader("Calibration and Performance Monitoring")
    st.write(
        "Continuous calibration monitoring is not currently implemented. Organizations considering prospective use "
        "should evaluate discrimination, calibration, screening yield, missing-data patterns, and subgroup performance "
        "using local data. Unexpected changes should trigger review and possible suspension of use."
    )

    st.subheader("Clinical Responsibility")
    st.warning(
        "Results must be interpreted by appropriately trained health professionals. The tool does not replace clinical "
        "judgment, validated sarcopenia assessments, local referral protocols, or urgent evaluation when clinically indicated."
    )

    st.subheader("Language Availability")
    st.write(
        "The current version is available in English only. A Korean-language version has not yet undergone professional "
        "translation, usability testing, or validation."
    )

    st.subheader("Documentation and Change History")
    st.markdown(
        f"Model version {MODEL_VERSION} | Updated {LAST_UPDATED} | "
        f"[Source code and version history]({REPOSITORY_URL})"
    )
