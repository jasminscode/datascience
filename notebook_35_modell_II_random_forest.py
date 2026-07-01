# %% [markdown]
# # Notebook 35 – Random Forest Regressor (Modell II)
#
# **Projekt:** Wirtschaftlichkeit von Airbnb-Listings in Spanien
# **Autorin:** Mareike Stahl (951648)
#
# ---
#
# ## Struktur dieses Notebooks
#
# 1. Daten laden & vorbereiten (lädt `cache/20_features_engineered.csv`)
# 2. Train/Test Split
# 3. Baseline Random Forest (Default-Parameter)
# 4. Hyperparameter Tuning (RandomizedSearchCV)
# 5. Bestes Modell evaluieren
# 6. Visualisierungen
# 7. Zusammenfassung für das Paper (Abschnitt 2.6.2 / 2.6.3)

# %% [markdown]
# ## 0 – Imports & Konfiguration

# %%
"""
Notebook 35 - Random Forest Regressor (Modell II)

VORAUSSETZUNG:
    notebook_30_feature_engineering.py muss vorher ausgeführt
    worden sein, damit cache/20_features_engineered.csv existiert.

AUSFÜHREN:
    python notebook_35_random_forest.py

AUSGABE:
    - Modell-Plots in: output_plots/rf_*.png
    - Ergebnisse werden im Terminal ausgegeben
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "cache/20_features_engineered.csv"
PLOT_DIR  = "output_plots"
SEED      = 42

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# %% [markdown]
# ## 1 – Daten laden & vorbereiten

# %%
print("=" * 60)
print("SCHRITT 1: Daten laden & vorbereiten")
print("=" * 60)

df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  Datensatz: {len(df):,} Zeilen, {df.shape[1]} Spalten")

FEATURE_COLS = [
    "dist_km_madrid_centro",
    "dist_km_barcelona_centro",
    "dist_km_sevilla_centro",
    "dist_km_malaga_centro",
    "dist_km_palma_centro",
    "dist_km_valencia_centro",
    "dist_km_bilbao_centro",
    "dist_km_girona_centro",
    "dist_km_nearest_hotspot",
    "room_Entire_home_apt",
    "room_Hotel_room",
    "room_Private_room",
    "room_Shared_room",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "number_of_reviews_ltm",
    "occupancy_rate",
    "region_encoded",
]

TARGET = "price"

df_model = df[FEATURE_COLS + [TARGET]].dropna()
print(f"  Nach Entfernung fehlender Werte: {len(df_model):,} Zeilen")

df_model = df_model[df_model[TARGET] <= 600]
print(f"  Nach Ausreißerfilter (Preis ≤ 600 €): {len(df_model):,} Zeilen")

X = df_model[FEATURE_COLS]
y = df_model[TARGET]

print(f"\n  Features ({len(FEATURE_COLS)}):")
for f in FEATURE_COLS:
    print(f"    - {f}")
print(f"\n  Zielvariable: {TARGET}")
print(f"  Preis – Min: {y.min():.0f} €, Max: {y.max():.0f} €, Median: {y.median():.0f} €")

# %% [markdown]
# ## 2 – Train/Test Split (80/20)

# %%
print("\n" + "=" * 60)
print("SCHRITT 2: Train/Test Split (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"  Trainingsdaten:  {len(X_train):,} Zeilen")
print(f"  Testdaten:       {len(X_test):,} Zeilen")

# %% [markdown]
# ## 3 – Baseline Random Forest (Default-Parameter)

# %%
print("\n" + "=" * 60)
print("SCHRITT 3: Baseline Random Forest (Default-Parameter)")
print("=" * 60)

rf_baseline = RandomForestRegressor(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)
print("  Trainiere Baseline-Modell... (kann 1-2 Minuten dauern)")
rf_baseline.fit(X_train, y_train)

y_pred_baseline = rf_baseline.predict(X_test)

mae_base  = mean_absolute_error(y_test, y_pred_baseline)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_base   = r2_score(y_test, y_pred_baseline)

print(f"\n  Baseline Ergebnisse:")
print(f"    MAE:  {mae_base:.2f} €")
print(f"    RMSE: {rmse_base:.2f} €")
print(f"    R²:   {r2_base:.4f}")

# %% [markdown]
# ## 4 – Hyperparameter Tuning (RandomizedSearchCV)
#
# RandomizedSearch statt GridSearch, da der Suchraum mit GridSearch
# sehr lange dauern würde. 20 zufällige Kombinationen mit 3-facher
# Cross-Validation sind ein guter Kompromiss aus Laufzeit und Qualität.

# %%
print("\n" + "=" * 60)
print("SCHRITT 4: Hyperparameter Tuning (RandomizedSearchCV)")
print("=" * 60)

param_dist = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.5],
}

rf_tuning = RandomForestRegressor(random_state=SEED, n_jobs=-1)

search = RandomizedSearchCV(
    rf_tuning,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring="r2",
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

print("  Starte Hyperparameter-Suche... (kann 3-5 Minuten dauern)")
search.fit(X_train, y_train)

print(f"\n  Beste Parameter:")
for k, v in search.best_params_.items():
    print(f"    {k}: {v}")
print(f"\n  Bester CV R²-Score: {search.best_score_:.4f}")

# %% [markdown]
# ## 5 – Bestes Modell evaluieren

# %%
print("\n" + "=" * 60)
print("SCHRITT 5: Bestes Modell evaluieren")
print("=" * 60)

rf_best = search.best_estimator_
y_pred_best = rf_best.predict(X_test)

mae_best  = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best   = r2_score(y_test, y_pred_best)

print(f"\n  Tuned Modell Ergebnisse:")
print(f"    MAE:  {mae_best:.2f} €")
print(f"    RMSE: {rmse_best:.2f} €")
print(f"    R²:   {r2_best:.4f}")

print(f"\n  Verbesserung durch Tuning:")
print(f"    MAE:  {mae_base:.2f} → {mae_best:.2f} € (Δ {mae_base - mae_best:+.2f} €)")
print(f"    RMSE: {rmse_base:.2f} → {rmse_best:.2f} € (Δ {rmse_base - rmse_best:+.2f} €)")
print(f"    R²:   {r2_base:.4f} → {r2_best:.4f} (Δ {r2_best - r2_base:+.4f})")

# %% [markdown]
# ## 6 – Visualisierungen

# %% [markdown]
# ### 6.1 – Vorhergesagt vs. Tatsächlich

# %%
print("\n" + "=" * 60)
print("SCHRITT 6: Visualisierungen erstellen")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Random Forest – Vorhergesagt vs. Tatsächlich", fontsize=13)

for ax, y_pred, title in zip(
    axes,
    [y_pred_baseline, y_pred_best],
    ["Baseline (Default)", f"Tuned (R²={r2_best:.3f})"]
):
    ax.scatter(y_test, y_pred, alpha=0.15, s=5, color="#4C72B0")
    lims = [0, 600]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfekte Vorhersage")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Tatsächlicher Preis (€)")
    ax.set_ylabel("Vorhergesagter Preis (€)")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.tight_layout()
path = f"{PLOT_DIR}/rf_predicted_vs_actual.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 6.2 – Residuen-Plot

# %%
residuals = y_test - y_pred_best

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Random Forest – Residuenanalyse (Tuned Modell)", fontsize=13)

ax1.scatter(y_pred_best, residuals, alpha=0.15, s=5, color="#DD8452")
ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax1.set_xlabel("Vorhergesagter Preis (€)")
ax1.set_ylabel("Residuum (€)")
ax1.set_title("Residuen vs. Vorhersage")

ax2.hist(residuals, bins=60, color="#55A868", edgecolor="white", linewidth=0.4)
ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Residuum (€)")
ax2.set_ylabel("Anzahl")
ax2.set_title(f"Residuenverteilung (Mean: {residuals.mean():.2f} €)")

plt.tight_layout()
path = f"{PLOT_DIR}/rf_residuals.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 6.3 – Feature Importance

# %%
importances = pd.Series(
    rf_best.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#4C72B0" if v >= importances.quantile(0.75) else "#a8c4e0"
          for v in importances]
importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Feature Importance – Random Forest (Tuned)", fontsize=13)
ax.set_xlabel("Importance (Mean Decrease Impurity)")
ax.axvline(importances.mean(), color="red", linestyle="--",
           linewidth=1, label=f"Durchschnitt ({importances.mean():.3f})")
ax.legend(fontsize=9)
plt.tight_layout()
path = f"{PLOT_DIR}/rf_feature_importance.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 6.4 – Modellvergleich (Baseline vs. Tuned)

# %%
metrics = {
    "MAE (€)":  [mae_base,  mae_best],
    "RMSE (€)": [rmse_base, rmse_best],
    "R²":       [r2_base,   r2_best],
}
labels = ["Baseline", "Tuned"]
colors_bar = ["#a8c4e0", "#4C72B0"]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Modellvergleich: Baseline vs. Tuned Random Forest", fontsize=13)

for ax, (metric, values) in zip(axes, metrics.items()):
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", width=0.5)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    better_idx = 1 if metric == "R²" else (1 if values[1] < values[0] else 0)
    bars[better_idx].set_edgecolor("#2d5a8e")
    bars[better_idx].set_linewidth(2)

plt.tight_layout()
path = f"{PLOT_DIR}/rf_model_comparison.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 6.5 – Top 10 Feature Importance (für Paper)

# %%
top10 = importances.tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
top10.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Top 10 wichtigste Features – Random Forest", fontsize=13)
ax.set_xlabel("Feature Importance")
plt.tight_layout()
path = f"{PLOT_DIR}/rf_top10_features.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

print("\n  → Alle Plots fertig gespeichert in: output_plots/")

# %% [markdown]
# ## 7 – Zusammenfassung für das Paper (Abschnitt 2.6.2 / 2.6.3)

# %%
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG (für Abschnitt 2.6.2 im Paper)")
print("=" * 60)

top5 = importances.tail(5).index.tolist()[::-1]

print(f"""
  Modell: Random Forest Regressor
  Bibliothek: scikit-learn

  Daten:
    Trainingsdaten: {len(X_train):,} Listings
    Testdaten:      {len(X_test):,} Listings
    Features:       {len(FEATURE_COLS)}

  Beste Hyperparameter:
""")
for k, v in search.best_params_.items():
    print(f"    {k}: {v}")

print(f"""
  Ergebnisse:
                  Baseline    Tuned
    MAE (€):      {mae_base:>8.2f}   {mae_best:>8.2f}
    RMSE (€):     {rmse_base:>8.2f}   {rmse_best:>8.2f}
    R²:           {r2_base:>8.4f}   {r2_best:>8.4f}

  Top 5 wichtigste Features:
""")
for i, f in enumerate(top5, 1):
    imp = importances[f]
    print(f"    {i}. {f}: {imp:.4f}")

print(f"""
  Interpretation:
    Der Random Forest Regressor erklärt {r2_best*100:.1f}% der Preisvarianz.
    Der durchschnittliche Vorhersagefehler beträgt {mae_best:.0f} €.
    Die wichtigsten Preistreiber sind geografische Lage und Zimmertyp.
""")

print("=" * 60)
print("  FERTIG! Notebook 35 abgeschlossen.")
print("  → Plots in output_plots/rf_*.png")
print("  → Ergebnisse oben für Abschnitt 2.6.2 verwenden")
print("=" * 60)
