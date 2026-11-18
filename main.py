
# ==========================================
# STUDENT PERFORMANCE PREDICTION
# SWYNEX - FINAL DATA SCIENCE PROJECT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/student-mat.csv", sep=";")

print("\n========== DATASET OVERVIEW ==========")
print("\nFirst 5 Student Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 2. DATA PREPARATION
# ==========================================

df = df.drop_duplicates()

df["performance"] = df["G3"].apply(
    lambda grade: "Pass" if grade >= 10 else "At Risk"
)

print("\n========== PERFORMANCE ANALYSIS ==========")
print(df["performance"].value_counts())

print("\nPerformance Percentage:")
print(
    (df["performance"].value_counts(normalize=True) * 100).round(2)
)

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

os.makedirs("graphs", exist_ok=True)
sns.set_theme(style="whitegrid")

# GRAPH 1: Final Grade Distribution

plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=10, kde=True)
plt.title("Distribution of Student Final Grades")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("graphs/grade_distribution.png", dpi=300)

# GRAPH 2: Study Time vs Final Grades

plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grades")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("graphs/studytime_vs_grades.png", dpi=300)

# GRAPH 3: Absences vs Final Grades

plt.figure(figsize=(8, 5))
sns.scatterplot(x="absences", y="G3", data=df)
plt.title("Absences vs Final Grades")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("graphs/absences_vs_grades.png", dpi=300)

# GRAPH 4: Pass vs At Risk Students

plt.figure(figsize=(7, 5))
sns.countplot(x="performance", data=df)
plt.title("Student Performance Distribution")
plt.xlabel("Performance Category")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("graphs/performance_distribution.png", dpi=300)

# GRAPH 5: Correlation Heatmap

plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Student Data Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", dpi=300)

# ==========================================
# 4. MACHINE LEARNING MODEL BUILDING
# ==========================================

print("\n========== MACHINE LEARNING MODEL ==========")

# Target variable
y = df["performance"]

# Remove G1, G2 and G3 to avoid data leakage
X = df.drop(
    columns=["G1", "G2", "G3", "performance"]
)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:", numeric_features)
print("\nCategorical Features:", categorical_features)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Random Forest Model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# ==========================================
# 5. TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# ==========================================
# 6. MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["At Risk", "Pass"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["At Risk", "Pass"]
)

fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=300)

# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/student_performance_model.pkl"
)

print("\nTrained model saved successfully!")

# ==========================================
# 9. SAVE PREPARED DATASET
# ==========================================

df.to_csv(
    "data/prepared_student_data.csv",
    index=False
)

print("Prepared dataset saved successfully!")

# ==========================================
# 10. FINAL PROJECT STATUS
# ==========================================

print("\n========== FINAL PROJECT STATUS ==========")
print("Dataset Loading: Completed")
print("Data Preparation: Completed")
print("EDA: Completed")
print("Machine Learning: Completed")
print("Model Evaluation: Completed")
print("Model Saved: Completed")
print("\nAll graphs saved in the graphs folder.")
print("Displaying graphs...")

# Display all graphs together
plt.show()
