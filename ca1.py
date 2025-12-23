# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ================= LOAD DATA =================
df = pd.read_csv(r"C:\Users\Asus\Desktop\Pred. Ca\bank\bank.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())


# ================= DATA PREPROCESSING =================
print("\nMissing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

# Encode categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# ================= SIMPLE EDA =================
plt.figure(figsize=(6,4))
sns.countplot(x='deposit', data=df)
plt.title("Deposit Subscription Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='deposit', y='duration', data=df)
plt.title("Call Duration vs Deposit")
plt.show()

# Simple and clean correlation heatmap
important_features = [
    'deposit',
    'age',
    'balance',
    'duration',
    'campaign',
    'previous'
]

corr = df[important_features].corr()

plt.figure(figsize=(6,4))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    cbar=False
)
plt.title("Correlation Heatmap (Key Features)")
plt.tight_layout()
plt.show()


# ================= FEATURE SELECTION =================
X = df.drop('deposit', axis=1)
y = df['deposit']


# ================= TRAIN-TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ================= MODEL 1: LOGISTIC REGRESSION =================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# ================= MODEL 2: DECISION TREE =================
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# ================= MODEL 3: KNN =================
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))


# ================= MODEL 4: NAIVE BAYES =================
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

print("\nNAIVE BAYES")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))


# ================= MODEL 5: SUPPORT VECTOR MACHINE =================
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\nSUPPORT VECTOR MACHINE")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))


# ================= MODEL COMPARISON =================
summary = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Naive Bayes',
        'SVM'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm)
    ]
})

print("\nFINAL MODEL COMPARISON")
print(summary)

summary.set_index('Model').plot(kind='bar', figsize=(8,4))
plt.ylabel("Accuracy")
plt.title("Supervised Classification Model Comparison")
plt.show()


# ================= CROSS VALIDATION =================
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_train_scaled,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Accuracy (Logistic Regression):", cv_scores.mean())
