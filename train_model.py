# ============================================================
# BANK MARKETING PREDICTION
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASETS
# ============================================================

# IMPORTANT:
# Your CSV files use ; instead of ,
train_df = pd.read_csv(
    "train (11).csv",
    sep=";"
)

test_df = pd.read_csv(
    "test (4).csv",
    sep=";"
)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

train_df.columns = (
    train_df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)

test_df.columns = (
    test_df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)


# ============================================================
# 3. DISPLAY COLUMNS
# ============================================================

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())


# ============================================================
# 4. CHECK TARGET
# ============================================================

print("\nTarget values in train:")
print(train_df["y"].value_counts())

print("\nTarget values in test:")
print(test_df["y"].value_counts())


# ============================================================
# 5. COMBINE TRAIN + TEST
# ============================================================

df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

print("\nCombined dataset shape:")
print(df.shape)


# ============================================================
# 6. SAVE COMBINED DATASET
# ============================================================

# Save using ; so the structure remains correct
df.to_csv(
    "bank.csv",
    sep=";",
    index=False
)

print("\nCombined dataset saved as bank.csv")


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

print("\nDuplicates before:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:")
print(df.duplicated().sum())


# ============================================================
# 8. CONVERT TARGET
# ============================================================

df["y"] = (
    df["y"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["y"] = df["y"].map({
    "yes": 1,
    "no": 0
})


# ============================================================
# 9. CHECK TARGET AFTER CONVERSION
# ============================================================

print("\nTarget after conversion:")
print(df["y"].value_counts())


# ============================================================
# 10. SEPARATE X AND y
# ============================================================

X = df.drop(
    "y",
    axis=1
)

y = df["y"]


print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 11. FIND CATEGORICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 12. FIND NUMERICAL COLUMNS
# ============================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# 13. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ============================================================
# 14. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ============================================================
# 15. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 16. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n======================================")
print("TRAINING RANDOM FOREST")
print("======================================")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# 18. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 19. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("MODEL RESULTS")
print("======================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "model.pkl",
    compress=3
)
print("\n======================================")
print("SUCCESS!")
print("======================================")

print("model.pkl created successfully!")
print("bank.csv created successfully!")
