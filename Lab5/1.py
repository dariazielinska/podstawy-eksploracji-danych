# Zadanie 1: Analiza istotności cech i standaryzacja

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
import os

# Upewnij się, że folder na wykresy istnieje
os.makedirs("figures", exist_ok=True)

# Wczytanie danych
diabetes_data = load_diabetes()
diabetes = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
diabetes['target'] = diabetes_data.target

# --- Wstępna analiza danych ---
print("Informacje o danych:")
print(diabetes.info())
print("\nOpis statystyczny:")
print(diabetes.describe())

# Sprawdzenie brakujących danych
missing = diabetes.isnull().sum()
print("\nBrakujące dane:")
print(missing)

# Wizualizacja rozkładów cech
plt.figure(figsize=(16, 12))
for i, column in enumerate(diabetes.columns[:-1]):
    plt.subplot(4, 3, i + 1)
    sns.histplot(diabetes[column], kde=True)
    plt.title(f'Rozkład: {column}')
plt.tight_layout()
plt.savefig("figures/feature_distributions.png")
plt.close()

# --- Standaryzacja danych ---
X = diabetes.drop('target', axis=1)
y = diabetes['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# --- Fisher Score (tu: korelacja cech z targetem) ---
correlations = [np.corrcoef(X_scaled_df[col], y)[0, 1] for col in X.columns]
fisher_df = pd.DataFrame({
    'Feature': X.columns,
    'Correlation_with_Target': correlations
}).sort_values(by='Correlation_with_Target', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation_with_Target', y='Feature', data=fisher_df)
plt.title('Współczynnik korelacji cech z targetem (Fisher-like interpretacja)')
plt.tight_layout()
plt.savefig("figures/fisher_like_scores.png")
plt.close()

print("\nFisher-like Scores (korelacja cech z targetem):")
print(fisher_df)

# --- Metoda ANOVA F-test ---
f_values, p_values = f_regression(X_scaled_df, y)

anova_results = pd.DataFrame({
    'Feature': X.columns,
    'F_value': f_values,
    'p_value': p_values
}).sort_values(by='F_value', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='F_value', y='Feature', data=anova_results)
plt.title('ANOVA F-score dla każdej cechy')
plt.tight_layout()
plt.savefig("figures/anova_f_scores.png")
plt.close()

print("\nWyniki ANOVA F-test:")
print(anova_results)

# --- Metoda MRMR ---

def mrmr_selection(X, y, k):
    """
    Uproszczona wersja MRMR oparta na korelacji.
    X: DataFrame ze standaryzowanymi cechami
    y: target (Series)
    k: liczba cech do wybrania
    """
    # Relevance: korelacja cech z targetem
    relevance = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))

    selected = []
    remaining = list(X.columns)

    for _ in range(k):
        mrmr_scores = []
        for feature in remaining:
            rel = relevance[feature]
            if selected:
                red = np.mean([abs(np.corrcoef(X[feature], X[sel])[0, 1]) for sel in selected])
            else:
                red = 0
            score = rel - red
            mrmr_scores.append((feature, score))
        
        best_feature = max(mrmr_scores, key=lambda x: x[1])[0]
        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected

# --- MRMR (prosta wersja bazująca na korelacji) ---
selected_mrmr = mrmr_selection(X_scaled_df, y, k=5)

print("\nTop 5 cech wybranych uproszczoną metodą MRMR:")
print(selected_mrmr)

# --- Model 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split

def evaluate_model(X_subset, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'f1_score': f1,
            'mcc': mcc
        }

    print(f"\n== Wyniki dla: {name} ==")
    print("DANE TRENINGOWE:")
    print(get_metrics(y_train, y_pred_train))
    print("DANE TESTOWE:")
    print(get_metrics(y_test, y_pred_test))

# === Korekta targetu do klasyfikacji (np. binarnej: > mediany to 1, inaczej 0)
y_binary = (y > y.median()).astype(int)

# === ANOVA top-5 cech
top_anova = anova_results.head(5)['Feature'].tolist()
evaluate_model(X_scaled_df[top_anova], y_binary, "ANOVA")

# === Fisher top-5 cech (korelacja)
top_fisher = fisher_df.head(5)['Feature'].tolist()
evaluate_model(X_scaled_df[top_fisher], y_binary, "Fisher-like")

# === MRMR top-5 cech
evaluate_model(X_scaled_df[selected_mrmr], y_binary, "MRMR")

# === Wszystkie cechy
evaluate_model(X_scaled_df, y_binary, "Wszystkie cechy")

