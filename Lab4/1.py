import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Wczytanie danych
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
medinc = df['MedInc']

mean = medinc.mean()
std = medinc.std()
lower_bound = mean - 3 * std
upper_bound = mean + 3 * std

outliers_std = medinc[(medinc < lower_bound) | (medinc > upper_bound)]
print(f"Liczba wartości odstających (3 sigma): {len(outliers_std)}")

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(medinc, bins=50, kde=True)
plt.axvline(lower_bound, color='red', linestyle='--', label='Dolny próg (3σ)')
plt.axvline(upper_bound, color='red', linestyle='--', label='Górny próg (3σ)')
plt.legend()
plt.title("Wartości odstające wg 3-sigma")
plt.savefig("outliers_3sigma_hist.png")
plt.show()

# Usunięcie i potwierdzenie
df_std_cleaned = df[(medinc >= lower_bound) & (medinc <= upper_bound)]

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_std_cleaned['MedInc'])
plt.title("Boxplot po usunięciu 3-sigma outliers")
plt.savefig("outliers_3sigma_boxplot.png")
plt.show()

Q1 = medinc.quantile(0.25)
Q3 = medinc.quantile(0.75)
IQR = Q3 - Q1
lower_bound_iqr = Q1 - 1.5 * IQR
upper_bound_iqr = Q3 + 1.5 * IQR

outliers_iqr = medinc[(medinc < lower_bound_iqr) | (medinc > upper_bound_iqr)]
print(f"Liczba wartości odstających (IQR): {len(outliers_iqr)}")

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(medinc, bins=50, kde=True)
plt.axvline(lower_bound_iqr, color='purple', linestyle='--', label='Dolny próg (IQR)')
plt.axvline(upper_bound_iqr, color='purple', linestyle='--', label='Górny próg (IQR)')
plt.legend()
plt.title("Wartości odstające wg IQR")
plt.savefig("outliers_IQR_hist.png")
plt.show()

# Usunięcie i potwierdzenie
df_iqr_cleaned = df[(medinc >= lower_bound_iqr) & (medinc <= upper_bound_iqr)]

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_iqr_cleaned['MedInc'])
plt.title("Boxplot po usunięciu IQR outliers")
plt.savefig("outliers_IQR_boxplot.png")
plt.show()

p01 = medinc.quantile(0.01)
p99 = medinc.quantile(0.99)

outliers_pct = medinc[(medinc < p01) | (medinc > p99)]
print(f"Liczba wartości odstających (percentyle 1% i 99%): {len(outliers_pct)}")

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(medinc, bins=50, kde=True)
plt.axvline(p01, color='green', linestyle='--', label='1. percentyl')
plt.axvline(p99, color='green', linestyle='--', label='99. percentyl')
plt.legend()
plt.title("Wartości odstające wg percentyli")
plt.savefig("outliers_percentiles_hist.png")
plt.show()

# Usunięcie i potwierdzenie
df_pct_cleaned = df[(medinc >= p01) & (medinc <= p99)]

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_pct_cleaned['MedInc'])
plt.title("Boxplot po usunięciu percentylowych outliers")
plt.savefig("outliers_percentiles_boxplot.png")
plt.show()
