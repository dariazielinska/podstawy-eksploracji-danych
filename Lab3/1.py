import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from scipy.stats import shapiro, pearsonr, spearmanr, skew
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytanie zbioru danych i konwersja do DataFrame
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# 2. Obliczenie podstawowych miar statystycznych
stats = pd.DataFrame({
    'mean': df.mean(),
    'median': df.median(),
    'std': df.std(),
    'var': df.var(),
    '25%': df.quantile(0.25),
    '75%': df.quantile(0.75),
    'skewness': df.skew()
})

# Zapis do pliku CSV
stats.to_csv("wine_basic_stats.csv")

# 3. Testy normalności Shapiro-Wilka
normality_results = {}
for col in df.columns[:-1]:  # bez target
    stat, p = shapiro(df[col])
    normality_results[col] = {"statistic": stat, "p_value": p, "normal": p > 0.05}

normality_df = pd.DataFrame(normality_results).T

# 4. Korelacja (Pearson lub Spearman zależnie od normalności)
corr_matrix = pd.DataFrame(index=df.columns[:-1], columns=df.columns[:-1])
p_values = pd.DataFrame(index=df.columns[:-1], columns=df.columns[:-1])

for col1 in df.columns[:-1]:
    for col2 in df.columns[:-1]:
        if normality_df.loc[col1]['normal'] and normality_df.loc[col2]['normal']:
            corr, p = pearsonr(df[col1], df[col2])
        else:
            corr, p = spearmanr(df[col1], df[col2])
        corr_matrix.loc[col1, col2] = corr
        p_values.loc[col1, col2] = p

# Zapis korelacji i p-value do pliku
corr_matrix.to_csv("wine_correlation_matrix.csv")
p_values.to_csv("wine_p_values.csv")

# 5. Wizualizacja mapy ciepła korelacji
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Macierz korelacji cech chemicznych w winie")
plt.tight_layout()
plt.savefig("wine_correlation_heatmap.png")
plt.show()
