import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 

#part 1
df = pd.read_csv(r"C:\Users\Maria\Downloads\Telegram Desktop\healthcare-dataset-stroke-data.csv")
print(df.head())
df.info()
print(df.describe())

#part 2
plt.figure(figsize=(5, 4))
sns.countplot(x="stroke", data=df)
plt.title("Stroke Class Distribution")
plt.show()
print(df["stroke"].value_counts(normalize=True))

print("Gender counts before:", df["gender"].value_counts().to_dict())
df = df[df["gender"] != "Other"].reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ["gender", "work_type", "smoking_status"]):
    sns.countplot(x=col, hue="stroke", data=df, ax=ax)
    ax.set_title(f"{col} vs stroke")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ["age", "avg_glucose_level", "bmi"]):
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

num_cols = ["age", "hypertension", "heart_disease",
            "avg_glucose_level", "bmi", "stroke"]
plt.figure(figsize=(7, 5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#part 3
print(df.isnull().sum())

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

print("Missing:", df.isnull().sum().sum())

print("Duplicates:", df.drop(columns="id").duplicated().sum())
df = df.drop_duplicates(subset=df.columns.difference(["id"]))
df = df.drop(columns="id")

plt.figure(figsize=(7, 3))
sns.boxplot(x=df["avg_glucose_level"])
plt.title("avg_glucose_level - before outlier removal")
plt.show()

REMOVE_GLUCOSE_OUTLIERS = True 
if REMOVE_GLUCOSE_OUTLIERS:
    q1, q3 = df["avg_glucose_level"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[df["avg_glucose_level"].between(low, high)]
    print(f"Removed {before - len(df)} outliers (bounds: {low:.1f} - {high:.1f})")
    print("Stroke rate after removal:", df["stroke"].mean().round(4))

#part 4
df["ever_married"] = df["ever_married"].map({"No": 0, "Yes": 1})
df["Residence_type"] = df["Residence_type"].map({"Rural": 0, "Urban": 1})
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

df = df.reset_index(drop=True)

df = pd.get_dummies(df, columns=["work_type", "smoking_status"], drop_first=True)
bool_cols = df.select_dtypes("bool").columns
df[bool_cols] = df[bool_cols].astype(int)
print(df.head())
print(df.dtypes)

#part 5
X = df.drop(columns="stroke")
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

#part 6
scale_cols = ["age", "avg_glucose_level", "bmi"]
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
print(X_train[scale_cols].describe().loc[["mean", "std"]])

#part 7
print("Before SMOTE:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", y_train_res.value_counts())
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x=y_train, ax=axes[0]).set_title("Before SMOTE")
sns.countplot(x=y_train_res, ax=axes[1]).set_title("After SMOTE")
plt.show()