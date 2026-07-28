"""
Agricultural plastic cover classification from satellite-derived features.

Ported from an original Google Earth Engine JavaScript pipeline (Random
Forest via ee.Classifier.smileRandomForest) to Python/scikit-learn.
Trains one Random Forest across all three regions (Kenya, Spain, Vietnam)
pooled together and produces region-prefixed submission IDs matching the
competition's expected format (e.g. 'Kenya_1', 'Spain_42', 'VNM_103').

See README.md for background, index definitions, and results.
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data_dir = os.path.join(os.path.dirname(__file__), "data")

# ---- Load + region-prefix IDs -------------------------------------------
# Raw IDs restart at 1 within each region's file, so they collide across
# regions (Kenya row 1, Spain row 1, VNM row 1 would all be '1'). The
# competition's expected submission format is region-prefixed
# ('Kenya_1', 'Spain_1', ...) -- see data/SampleSubmission.csv.

kenya_train = pd.read_csv(os.path.join(data_dir, "Kenya_training.csv"))
kenya_train["ID"] = "Kenya_" + kenya_train["ID"].astype(str)

spain_train = pd.read_csv(os.path.join(data_dir, "Spain_training.csv"))
spain_train["ID"] = "Spain_" + spain_train["ID"].astype(str)

vnm_train = pd.read_csv(os.path.join(data_dir, "VNM_training.csv"))
vnm_train["ID"] = "VNM_" + vnm_train["ID"].astype(str)

training_data = pd.concat([kenya_train, spain_train, vnm_train], ignore_index=True)

kenya_test = pd.read_csv(os.path.join(data_dir, "Kenya_testing.csv"))
kenya_test["ID"] = "Kenya_" + kenya_test["ID"].astype(str)

vnm_test = pd.read_csv(os.path.join(data_dir, "VNM_testing.csv"))
vnm_test["ID"] = "VNM_" + vnm_test["ID"].astype(str)

spain_val = pd.read_csv(os.path.join(data_dir, "Spain_validation.csv"))
spain_val["ID"] = "Spain_" + spain_val["ID"].astype(str)

testing_data = pd.concat([kenya_test, vnm_test, spain_val], ignore_index=True)


# ---- Feature engineering --------------------------------------------------
# Spectral indices ported from the original GEE script (based on
# Aguilar et al.'s greenhouse/plastic-mapping approach). See README.md for
# the physical reasoning behind each index.

def compute_indices(df):
    df = df.copy()
    avg_bg_nir = (df["blue_p50"] + df["green_p50"] + df["nir_p50"]) / 3

    df["PMLI"] = (df["swir2_p50"] - df["red_p50"]) / (df["swir2_p50"] + df["red_p50"])
    df["RPGI"] = df["blue_p50"] / (avg_bg_nir - 1) * 100
    df["NDBI"] = (df["swir1_p50"] - df["nir_p50"]) / (df["swir1_p50"] + df["nir_p50"])
    df["VI"] = ((df["swir1_p50"] - df["nir_p50"]) / (df["swir1_p50"] + df["nir_p50"])) * \
               ((df["nir_p50"] - df["red_p50"]) / (df["nir_p50"] + df["red_p50"]))
    df["PGI"] = df["blue_p50"] * (df["nir_p50"] - df["red_p50"]) / (avg_bg_nir - 1) * 100
    df["NDVI"] = (df["nir_p50"] - df["red_p50"]) / (df["nir_p50"] + df["red_p50"])
    # Experimental: Plastic Greenhouse Index (swir2 / blue). Kept for
    # reference -- tested no more impactful than the indices above, not
    # removed from the feature set below in case it earns its place later.
    df["PGHI"] = df["swir2_p50"] / df["blue_p50"]
    return df


training_data = compute_indices(training_data)
testing_data = compute_indices(testing_data)

attributes = [
    "blue_p50", "green_p50", "nir_p50", "nira_p50", "re1_p50", "re2_p50",
    "re3_p50", "red_p50", "swir1_p50", "swir2_p50", "VV_p50", "VH_p50",
    "PMLI", "RPGI", "NDBI", "VI", "PGI", "NDVI",
]
target = "TARGET"

X = training_data[attributes]
y = training_data[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions on the actual competition test set, from the split-trained
# model above (not a separately-refit model -- one classifier throughout).
testing_data["classification"] = clf.predict(testing_data[attributes])

val_preds = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

submission = testing_data[["ID", "classification"]].rename(columns={"classification": "TARGET"})
submission.to_csv(os.path.join(data_dir, "submission.csv"), index=False)
print(f"Submission saved to {os.path.join(data_dir, 'submission.csv')}")
