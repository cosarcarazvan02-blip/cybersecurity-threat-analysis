import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42

df = pd.read_csv("cybersecurity_threat_dataset.csv")
print("First rows:")
print(df.head())

df = df.drop(columns=["timestamp", "ioc_value", "src_country"])

target_col = "label"
X = df.drop(columns=[target_col])
y = df[target_col]

if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

print("\nFirst 10 rows of X:")
print(X.head(10))

print("\nDescriptive statistics:")
print(X.describe(include="all"))
print("\nMedian (numeric columns):")
print(X.median(numeric_only=True))
print("\nStandard deviation (numeric columns):")
print(X.std(numeric_only=True))
print("\nUnique classes in y:", pd.Series(y).unique())
print("\nClass distribution:")
print(pd.Series(y).value_counts())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(x=pd.Series(y), ax=axes[0])
axes[0].set_title("Class distribution")
axes[0].set_xlabel("label")

X[["confidence_level", "dst_port", "days_active"]].hist(bins=20, ax=None, figsize=(12, 4))
plt.suptitle("Histograms of numeric features")

sns.heatmap(X.select_dtypes(include=np.number).corr(), annot=True, cmap="Blues", ax=axes[2])
axes[2].set_title("Correlation of numeric features")

plt.tight_layout()
plt.show()

X["confidence_cat"] = pd.cut(
    X["confidence_level"],
    bins=[-1, 33, 66, 100],
    labels=["low", "medium", "high"]
)

rng = np.random.RandomState(RANDOM_STATE)
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

mask = rng.rand(*X[numeric_cols].shape) < 0.05
X_num = X[numeric_cols].copy()
X_num = X_num.mask(mask)
X[numeric_cols] = X_num

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

models = {
    "KNN-5": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}

baseline_results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    baseline_results[name] = acc

    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\nBaseline model comparison:")
print(pd.Series(baseline_results).sort_values(ascending=False))

print("\nClass distribution before SMOTE:")
print(pd.Series(y).value_counts())

X_imb = X
y_imb = y

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=RANDOM_STATE, stratify=y_imb
)

pipe_smote = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
])

pipe_smote.fit(X_train_imb, y_train_imb)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
])

param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5]
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_model = grid.best_estimator_

train_sizes, train_scores, valid_scores = learning_curve(
    best_model,
    X_train_imb,
    y_train_imb,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=RANDOM_STATE
)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, label="Training accuracy", marker="o")
plt.plot(train_sizes, valid_mean, label="Validation accuracy", marker="o")
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.title("Learning curve")
plt.legend()
plt.tight_layout()
plt.show()

eval_X = X_test_imb
eval_y = y_test_imb
y_pred = best_model.predict(eval_X)

print("\nFinal accuracy:", accuracy_score(eval_y, y_pred))
print("\nClassification report:")
print(classification_report(eval_y, y_pred))

cm = confusion_matrix(eval_y, y_pred)
cm_all = cm / cm.sum() * 100
cm_row = cm / cm.sum(axis=1, keepdims=True) * 100
cm_col = cm / cm.sum(axis=0, keepdims=True) * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ["% of all samples", "Normalized by row (%)", "Normalized by column (%)"]
matrices = [cm_all, cm_row, cm_col]

for ax, matrix, title in zip(axes, matrices, titles):
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

pre = best_model.named_steps["preprocess"]
all_feature_names = pre.get_feature_names_out()

if hasattr(best_model.named_steps["clf"], "feature_importances_"):
    importances = best_model.named_steps["clf"].feature_importances_
elif hasattr(best_model.named_steps["clf"], "coef_"):
    importances = np.abs(best_model.named_steps["clf"].coef_).mean(axis=0)
else:
    importances = None

if importances is not None:
    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature",
                hue="feature", legend=False, palette="Blues_r")
    plt.title("Top 20 important features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

print("\n✅ Pipeline finished running successfully.")