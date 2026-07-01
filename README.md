# Wirtschaftlichkeit von Airbnb-Listings in Spanien

**Seminararbeit – IT Management M.Sc. | Hochschule Mainz**  
Annika Reiß (950644) · Mareike Stahl (951648) · Jasmin Müller (951624)  
Betreuer: Prof. Dr. Martin Huschens | Abgabe: 04. Juli 2026

---

## Projektstruktur

```
datascience/
│
├── data/
│   ├── raw/                              # Rohdaten je Region (Inside Airbnb)
│   │   ├── listings_barcelona.csv
│   │   ├── listings_euskadi.csv
│   │   ├── listings_girona.csv
│   │   ├── listings_madrid.csv
│   │   ├── listings_malaga.csv
│   │   ├── listings_mallorca.csv
│   │   ├── listings_menorca.csv
│   │   ├── listings_sevilla.csv
│   │   └── listings_valencia.csv
│   └── (listings_spanien_cleaned.csv)    # Output von Notebook 10
│
├── cache/
│   ├── 20_features_engineered.csv        # Output von Notebook 30 (Mareike)
│   ├── 40_metrics_model_III.csv          # Output von Notebook 40 (Jasmin)
│   ├── 40_feature_importance.csv
│   ├── 40_feature_names.csv
│   └── 40_model_III.pkl
│
├── output_plots/                         # EDA-Plots (Notebook 30)
├── outputs/                              # Modell-Plots (Notebook 40, 50)
│
├── notebook_10_datenaufbereitung.ipynb         # Annika:  Datenbereinigung & EDA
├── notebook_25_modell_I_poly_regression.ipynb  # Annika:  Modell I – Polynomiale Regression
├── notebook_30_feature_engineering.py          # Mareike: EDA & Feature Engineering
├── notebook_35_modell_II_random_forest.py      # Mareike: Modell II – Random Forest
├── notebook_40_model_III.ipynb                 # Jasmin:  Modell III – Decision Tree
├── notebook_50_evaluation.ipynb                # Jasmin:  Modellvergleich & Business Insights
│
├── requirements.txt
└── README.md
```

---

## Ausführungsreihenfolge (Pipeline)

```
Notebook 10  →  listings_spanien_cleaned.csv       (Annika: Datenbereinigung)
     ↓
Notebook 30  →  cache/20_features_engineered.csv   (Mareike: EDA + Feature Engineering)
     ↓
Notebook 25  →  Modell I:   Polynomiale Regression (Annika)
Notebook 35  →  Modell II:  Random Forest          (Mareike)
Notebook 40  →  Modell III: Decision Tree          (Jasmin)
     ↓
Notebook 50  →  Modellvergleich & Business Insights (Jasmin)
```

**Wichtig:** Notebook 30 muss vor den Modell-Notebooks (25, 35, 40) ausgeführt werden,
da alle drei Modelle auf demselben Feature-Datensatz `cache/20_features_engineered.csv` aufbauen.

---

## Setup & Ausführung

### 1. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 2. Pipeline ausführen

```bash
# Schritt 1: Datenbereinigung (Annika)
jupyter notebook notebook_10_datenaufbereitung.ipynb

# Schritt 2: Feature Engineering & EDA (Mareike) – erzeugt cache/20_features_engineered.csv
python notebook_30_feature_engineering.py

# Schritt 3: Modelle trainieren (parallel möglich, da alle auf gleichem Feature-Datensatz)
jupyter notebook notebook_25_modell_I_poly_regression.ipynb   # Annika
python notebook_35_modell_II_random_forest.py                  # Mareike
jupyter notebook notebook_40_model_III.ipynb                   # Jasmin

# Schritt 4: Vergleich (Jasmin)
jupyter notebook notebook_50_evaluation.ipynb
```

---

## Datensatz

| Merkmal | Wert |
|---|---|
| Quelle | Inside Airbnb – Spanien |
| Listings (roh) | ~99.000 |
| Regionen | Barcelona, Madrid, Girona, Mallorca, Malaga, Sevilla, Valencia, Euskadi, Menorca |
| Spalten (roh) | 19 |
| Spalten (nach Feature Engineering) | 36 |

---

## Modelle & Zuständigkeiten

| Notebook | Modell | Zuständig | Besonderheit |
|---|---|---|---|
| 10 | – | Annika | Datenbereinigung, EDA, 9 Regionen zusammenführen |
| 30 | – | Mareike | EDA-Visualisierungen, Haversine-Distanzen, OHE, `occupancy_rate` |
| 25 | Polynomiale Regression (Grad 1–2) | Annika | Baseline-Modell |
| 35 | Random Forest (tuned) | Mareike | RandomizedSearchCV, 21 Features |
| 40 | Decision Tree / GradientBoosting | Jasmin | GridSearchCV, nutzt Mareikes Feature-Datensatz |
| 50 | – | Jasmin | Modellvergleich, Residuenanalyse, Business Insights |

---

## Features (nach Engineering, Notebook 30)

| Feature | Beschreibung |
|---|---|
| `dist_km_*_centro` | Haversine-Distanz zu 8 Tourismus-Hotspots (km) |
| `dist_km_nearest_hotspot` | Distanz zum nächsten Hotspot |
| `room_Entire_home_apt` etc. | One-Hot: Zimmertyp |
| `estimated_revenue` | Preis × gebuchte Nächte (€/Jahr) |
| `occupancy_rate` | Anteil gebuchter Nächte (0–1) |
| `log_price` | log(1 + Preis) für Normalverteilung |
| `region_encoded` | Region als numerisches Label (0–8) |
