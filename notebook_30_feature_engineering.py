# %% [markdown]
# # Notebook 30 – EDA & Feature Engineering
#
# **Projekt:** Wirtschaftlichkeit von Airbnb-Listings in Spanien
# **Autorin:** Mareike Stahl (951648)
#
# ---
#
# ## Struktur dieses Notebooks
#
# 1. Daten laden
# 2. Vertiefte explorative Datenanalyse (Visualisierungen)
# 3. Feature Engineering (Hotspot-Distanzen, One-Hot-Encoding, abgeleitete Merkmale)
# 4. Speichern als `cache/20_features_engineered.csv`
# 5. Zusammenfassung für das Paper (Abschnitt 2.5)

# %% [markdown]
# ## 0 – Imports & Konfiguration

# %%
"""
Notebook 30 - EDA & Feature Engineering

SETUP (einmalig im Terminal ausführen):
    pip install pandas numpy matplotlib seaborn scikit-learn scipy

AUSFÜHREN:
    python notebook_30_feature_engineering.py

AUSGABE:
    - Alle Plots werden als PNG gespeichert (Ordner: output_plots/)
    - Bereinigter Feature-Datensatz: cache/20_features_engineered.csv
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Passe diesen Pfad an deinen lokalen Speicherort an:
INPUT_CSV  = "listings_spanien_cleaned.csv"   # <- dein Roh-CSV
OUTPUT_CSV = "cache/20_features_engineered.csv"
PLOT_DIR   = "output_plots"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)

# Plot-Stil
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

REGION_ORDER = [
    "Madrid", "Barcelona", "Girona", "Mallorca",
    "Malaga", "Sevilla", "Valencia", "Euskadi", "Menorca"
]

# %% [markdown]
# ## 1 – Daten laden

# %%
print("=" * 60)
print("SCHRITT 1: Daten laden")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"  Datensatz geladen: {len(df):,} Zeilen, {df.shape[1]} Spalten")
print(f"  Regionen: {sorted(df['region'].unique())}")
print(f"  Zimmertypen: {sorted(df['room_type'].unique())}")
print(f"\nVollständigkeit je Spalte:")
print((df.notna().sum() / len(df) * 100).round(1).to_string())

# %% [markdown]
# ## 2 – Vertiefte EDA: Visualisierungen
#
# Hier entstehen die Plots, die später in Abschnitt 2.4 des Papers verwendet werden.

# %%
print("\n" + "=" * 60)
print("SCHRITT 2: Explorative Datenanalyse (Visualisierungen)")
print("=" * 60)

# Nur Zeilen mit Preis für Preisplots
df_price = df[df["price"].notna()].copy()
df_clean = df_price[df_price["price"] <= 600].copy()  # Fokus-Bereich

# %% [markdown]
# ### 2.1 – Preisverteilung je Region (Histogramme)

# %%
# 2.1a – Alle Listings (ungeclipped) → zeigt Premiumsegment und Rechtsschieffe
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Preisverteilung je Region – alle Listings (ungeclipped)", fontsize=14, y=1.01)

for ax, region in zip(axes.flatten(), REGION_ORDER):
    data = df_price[df_price["region"] == region]["price"]
    n_total = len(data)
    n_premium = len(data[data > 600])
    ax.hist(data, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_title(f"{region}\n(n={n_total:,}, davon {n_premium} > 600 €)", fontsize=9)
    ax.set_xlabel("Preis (€)", fontsize=8)
    ax.set_ylabel("Anzahl", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
path = f"{PLOT_DIR}/2.1a_preisverteilung_alle.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# 2.1b – Geclipped bei 600€ → zeigt Hauptsegment detailliert
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Preisverteilung je Region – Hauptsegment (Preis ≤ 600 €)", fontsize=14, y=1.01)

for ax, region in zip(axes.flatten(), REGION_ORDER):
    data = df_clean[df_clean["region"] == region]["price"]
    n_total = len(df_price[df_price["region"] == region])
    n_hidden = n_total - len(data)
    ax.hist(data, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_title(f"{region}\n(n={n_total:,}, {n_hidden} > 600 € ausgeblendet)", fontsize=9)
    ax.set_xlabel("Preis (€)", fontsize=8)
    ax.set_ylabel("Anzahl", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
path = f"{PLOT_DIR}/2.1b_preisverteilung_600.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 2.2 – Preisboxplots je Region & Zimmertyp

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Preisboxplots je Region (Roh, ohne Extremwerte)", fontsize=13)

# Linkes Plot: nur nach Region
sns.boxplot(
    data=df_clean, y="region", x="price",
    order=REGION_ORDER, orient="h",
    palette="muted", fliersize=0, ax=ax1
)
ax1.set_title("Preis je Region")
ax1.set_xlabel("Preis (€)")
ax1.set_ylabel("Region")

# Rechtes Plot: Region + Zimmertyp
ROOM_PALETTE = {
    "Entire home/apt": "#4C72B0",
    "Private room":    "#DD8452",
    "Hotel room":      "#55A868",
    "Shared room":     "#C44E52",
}
sns.boxplot(
    data=df_clean, y="region", x="price",
    hue="room_type", order=REGION_ORDER, orient="h",
    palette=ROOM_PALETTE, fliersize=0, ax=ax2
)
ax2.set_title("Preis je Region & Zimmertyp")
ax2.set_xlabel("Preis (€)")
ax2.set_ylabel("")
ax2.legend(title="Zimmertyp", fontsize=8, loc="lower right")

plt.tight_layout()
path = f"{PLOT_DIR}/2.2_preisboxplots_region.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 2.3 – Zimmertyp-Anteile je Region

# %%
room_share = (
    df.groupby(["region", "room_type"])
    .size()
    .reset_index(name="count")
)
room_share["pct"] = room_share.groupby("region")["count"].transform(
    lambda x: x / x.sum() * 100
)

fig, ax = plt.subplots(figsize=(12, 6))
room_types_order = ["Entire home/apt", "Hotel room", "Private room", "Shared room"]
pivot = room_share.pivot(index="region", columns="room_type", values="pct").fillna(0)
pivot = pivot.reindex(REGION_ORDER)[room_types_order]
pivot.plot(kind="bar", ax=ax, color=list(ROOM_PALETTE.values()), edgecolor="white")
ax.set_title("Zimmertyp-Anteile je Region (%)", fontsize=13)
ax.set_xlabel("Region")
ax.set_ylabel("Anteil (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Zimmertyp", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
path = f"{PLOT_DIR}/2.3_zimmertyp_anteile.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 2.4 – Verfügbarkeit & Mindestaufenthalt

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analyse der Auslastung: Verfügbarkeit & Mindestaufenthalt", fontsize=13)

df_avail = df[df["availability_365"].notna()]
df_nights = df[(df["minimum_nights"].notna()) & (df["minimum_nights"] <= 365)]

axes[0, 0].hist(df_avail["availability_365"], bins=50, color="#4C72B0", edgecolor="white", lw=0.4)
axes[0, 0].axvline(df_avail["availability_365"].median(), color="red", linestyle="--",
                   label=f'Median: {df_avail["availability_365"].median():.0f} Tage')
axes[0, 0].set_title("Verfügbarkeit: Tage/Jahr")
axes[0, 0].set_xlabel("Verfügbare Tage")
axes[0, 0].set_ylabel("Anzahl Listings")
axes[0, 0].legend()

axes[0, 1].hist(df_nights["minimum_nights"], bins=50, color="#DD8452", edgecolor="white", lw=0.4)
axes[0, 1].set_title("Mindestaufenthalt (≤ 365 Tage)")
axes[0, 1].set_xlabel("Mindestaufenthalt (Tage)")
axes[0, 1].set_ylabel("Anzahl Listings")

sns.boxplot(data=df_avail, x="room_type", y="availability_365",
            palette=ROOM_PALETTE, fliersize=1, ax=axes[1, 0])
axes[1, 0].set_title("Verfügbarkeit nach Zimmertyp")
axes[1, 0].set_xlabel("Zimmertyp")
axes[1, 0].set_ylabel("Tage/Jahr verfügbar")
axes[1, 0].tick_params(axis="x", rotation=20)

sns.boxplot(data=df_nights, x="room_type", y="minimum_nights",
            palette=ROOM_PALETTE, fliersize=1, ax=axes[1, 1])
axes[1, 1].set_title("Mindestaufenthalt nach Zimmertyp")
axes[1, 1].set_xlabel("Zimmertyp")
axes[1, 1].set_ylabel("Mindestaufenthalt (Tage)")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.tight_layout()
path = f"{PLOT_DIR}/2.4_auslastung_verfuegbarkeit.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 2.5 – Bewertungsanalyse

# %%
# 2.5a – Allgemeine Bewertungsverteilung + nach Zimmertyp
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analyse der Bewertungen – Gesamtverteilung & Zimmertyp", fontsize=13)

df_rev = df[df["number_of_reviews"].notna()]
df_rpm = df[df["reviews_per_month"].notna()]

axes[0, 0].hist(df_rev["number_of_reviews"].clip(upper=500), bins=50,
                color="#55A868", edgecolor="white", lw=0.4)
med_rev = df_rev["number_of_reviews"].median()
axes[0, 0].axvline(med_rev, color="red", linestyle="--", label=f"Median: {med_rev:.0f}")
axes[0, 0].set_title("Verteilung: Anzahl Bewertungen (≤ 500)")
axes[0, 0].set_xlabel("Anzahl Bewertungen")
axes[0, 0].set_ylabel("Anzahl Listings")
axes[0, 0].legend()

axes[0, 1].hist(df_rpm["reviews_per_month"].clip(upper=20), bins=50,
                color="#8172B2", edgecolor="white", lw=0.4)
med_rpm = df_rpm["reviews_per_month"].median()
axes[0, 1].axvline(med_rpm, color="red", linestyle="--", label=f"Median: {med_rpm:.2f}")
axes[0, 1].set_title("Verteilung: Bewertungen/Monat (≤ 20)")
axes[0, 1].set_xlabel("Reviews pro Monat")
axes[0, 1].set_ylabel("Anzahl Listings")
axes[0, 1].legend()

sns.boxplot(data=df_rev, x="room_type", y="number_of_reviews",
            palette=ROOM_PALETTE, fliersize=1, ax=axes[1, 0])
axes[1, 0].set_ylim(0, 200)
axes[1, 0].set_title("Anzahl Bewertungen nach Zimmertyp")
axes[1, 0].tick_params(axis="x", rotation=20)

sns.boxplot(data=df_rpm, x="room_type", y="reviews_per_month",
            palette=ROOM_PALETTE, fliersize=1, ax=axes[1, 1])
axes[1, 1].set_ylim(0, 6)
axes[1, 1].set_title("Reviews/Monat nach Zimmertyp")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.tight_layout()
path = f"{PLOT_DIR}/2.5a_bewertungsanalyse.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# 2.5b – Bewertungsquote & monatliche Rate je Region
# Bewertungsquote = Anteil Listings mit mindestens einer Bewertung
region_review = df.groupby("region").agg(
    total=("id", "count"),
    mit_reviews=("number_of_reviews", lambda x: (x > 0).sum()),
    median_rpm=("reviews_per_month", "median"),
).reset_index()
region_review["quote_pct"] = (region_review["mit_reviews"] / region_review["total"] * 100).round(1)
region_review = region_review.sort_values("quote_pct", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Bewertungsaktivität je Region", fontsize=13)

# Linkes Plot: Bewertungsquote
colors_quote = ["#4C72B0" if q >= 88 else "#a8c4e0" for q in region_review["quote_pct"]]
bars = ax1.barh(region_review["region"], region_review["quote_pct"],
                color=colors_quote, edgecolor="white")
ax1.set_xlabel("Anteil Listings mit mind. 1 Bewertung (%)")
ax1.set_title("Bewertungsquote je Region")
ax1.set_xlim(70, 100)
for bar, val in zip(bars, region_review["quote_pct"]):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=9)

# Rechtes Plot: Median Reviews/Monat
region_review_sorted = region_review.sort_values("median_rpm", ascending=True)
colors_rpm = ["#4C72B0" if r >= 1.0 else "#a8c4e0" for r in region_review_sorted["median_rpm"]]
bars2 = ax2.barh(region_review_sorted["region"], region_review_sorted["median_rpm"],
                 color=colors_rpm, edgecolor="white")
ax2.set_xlabel("Median Reviews pro Monat")
ax2.set_title("Buchungsaktivität je Region (Reviews/Monat)")
for bar, val in zip(bars2, region_review_sorted["median_rpm"]):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center", fontsize=9)

plt.tight_layout()
path = f"{PLOT_DIR}/2.5b_bewertungsquote_region.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 2.6 – Korrelationsmatrix (preisrelevante Merkmale)

# %%
num_cols = [
    "price", "number_of_reviews", "reviews_per_month",
    "availability_365", "minimum_nights",
    "calculated_host_listings_count", "number_of_reviews_ltm"
]
corr_df = df_clean[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
    annot_kws={"size": 9}
)
ax.set_title("Korrelationsmatrix – Preisrelevante Merkmale (bereinigt)", fontsize=12)
plt.tight_layout()
path = f"{PLOT_DIR}/2.6_korrelationsmatrix.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

# %% [markdown]
# ### 2.7 – Preis vs. Verfügbarkeit / Bewertungen (Scatter)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Preiskorrelationen (bereinigter Datensatz)", fontsize=13)

sample = df_clean.dropna(subset=["price", "availability_365", "number_of_reviews"]).sample(
    min(5000, len(df_clean)), random_state=42
)
corr_avail = sample["price"].corr(sample["availability_365"])
corr_rev   = sample["price"].corr(sample["number_of_reviews"])

ax1.scatter(sample["price"], sample["number_of_reviews"],
            alpha=0.2, s=5, color="#4C72B0")
ax1.set_title(f"Preis vs. Anzahl Bewertungen\nKorrelation: {corr_rev:.3f}")
ax1.set_xlabel("Preis (€)")
ax1.set_ylabel("Anzahl Bewertungen")

ax2.scatter(sample["price"], sample["availability_365"],
            alpha=0.2, s=5, color="#DD8452")
ax2.set_title(f"Preis vs. Verfügbarkeit\nKorrelation: {corr_avail:.3f}")
ax2.set_xlabel("Preis (€)")
ax2.set_ylabel("Verfügbarkeit (Tage/Jahr)")

plt.tight_layout()
path = f"{PLOT_DIR}/2.7_preis_scatter.png"
plt.savefig(path)
plt.close()
print(f"  ✓ Gespeichert: {path}")

print("\n  → Alle Plots fertig gespeichert in: output_plots/")

# %% [markdown]
# ## 3 – Feature Engineering
#
# Hier entstehen die neuen Merkmale für Abschnitt 2.5 des Papers:
# Hotspot-Distanzen (Haversine), One-Hot-Encoding des Zimmertyps,
# sowie abgeleitete Größen wie Umsatzschätzung und Occupancy Rate.

# %%
print("\n" + "=" * 60)
print("SCHRITT 3: Feature Engineering")
print("=" * 60)

df_feat = df.copy()

# %% [markdown]
# ### 3.1 – Distanz zu Tourismus-Hotspots (Haversine-Formel)
#
# Berechnet die geodätische Distanz (km) von jedem Listing
# zu den wichtigsten touristischen Zentren Spaniens.

# %%
HOTSPOTS = {
    "dist_km_madrid_centro":    (40.4168, -3.7038),   # Puerta del Sol
    "dist_km_barcelona_centro": (41.3851,  2.1734),   # Las Ramblas
    "dist_km_sevilla_centro":   (37.3891, -5.9845),   # Catedral
    "dist_km_malaga_centro":    (36.7213, -4.4214),   # Alcazaba
    "dist_km_palma_centro":     (39.5696,  2.6502),   # Palma de Mallorca
    "dist_km_valencia_centro":  (39.4699, -0.3763),   # Plaza del Ayuntamiento
    "dist_km_bilbao_centro":    (43.2630, -2.9350),   # Guggenheim Museum
    "dist_km_girona_centro":    (41.9794,  2.8214),   # Girona Altstadt
}

def haversine_km(lat1, lon1, lat2, lon2):
    """Berechnet Distanz in km zwischen zwei GPS-Koordinaten."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Vektorisierte Version für Performance
def haversine_vec(lat, lon, lat2, lon2):
    R = 6371.0
    lat_r  = np.radians(lat)
    lon_r  = np.radians(lon)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dphi = lat2_r - lat_r
    dlam = lon2_r - lon_r
    a = np.sin(dphi / 2)**2 + np.cos(lat_r) * math.cos(lat2_r) * np.sin(dlam / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

print("\n  Berechne Distanzen zu Tourismus-Hotspots...")
for col_name, (lat2, lon2) in HOTSPOTS.items():
    mask = df_feat["latitude"].notna() & df_feat["longitude"].notna()
    df_feat.loc[mask, col_name] = haversine_vec(
        df_feat.loc[mask, "latitude"].values,
        df_feat.loc[mask, "longitude"].values,
        lat2, lon2
    )
    median_dist = df_feat.loc[mask, col_name].median()
    print(f"  ✓ {col_name}: Median = {median_dist:.1f} km")

# Distanz zum nächsten Hotspot (Min über alle 8)
hotspot_cols = list(HOTSPOTS.keys())
df_feat["dist_km_nearest_hotspot"] = df_feat[hotspot_cols].min(axis=1)
print(f"  ✓ dist_km_nearest_hotspot: Median = {df_feat['dist_km_nearest_hotspot'].median():.1f} km")

# %% [markdown]
# ### 3.2 – One-Hot-Encoding für room_type

# %%
print("\n  One-Hot-Encoding für room_type...")
room_dummies = pd.get_dummies(
    df_feat["room_type"],
    prefix="room",
    drop_first=False  # alle 4 Kategorien behalten (für Transparenz)
).astype(int)

# Spaltennamen bereinigen (Leerzeichen → Unterstriche)
room_dummies.columns = [c.replace(" ", "_").replace("/", "_") for c in room_dummies.columns]
print(f"  ✓ Neue Spalten: {list(room_dummies.columns)}")

df_feat = pd.concat([df_feat, room_dummies], axis=1)

# %% [markdown]
# ### 3.3 – Weitere abgeleitete Features

# %%
print("\n  Erstelle weitere abgeleitete Features...")

# Revenue-Schätzung: Preis × (365 - availability_365)
# = Preis × geschätzte gebuchte Nächte
df_feat["estimated_revenue"] = (
    df_feat["price"] * (365 - df_feat["availability_365"])
).clip(lower=0)
print(f"  ✓ estimated_revenue: Median = {df_feat['estimated_revenue'].median():.0f} €/Jahr")

# Occupancy Rate (0–1): Anteil gebuchter Nächte
df_feat["occupancy_rate"] = (
    (365 - df_feat["availability_365"]) / 365
).clip(0, 1)
print(f"  ✓ occupancy_rate: Median = {df_feat['occupancy_rate'].median():.2f}")

# Log-Preis (für normalverteilte Zielvariable im Modell)
df_feat["log_price"] = np.log1p(df_feat["price"])

# Region als numerische Kategorie (Label-Encoding)
region_map = {r: i for i, r in enumerate(sorted(df_feat["region"].dropna().unique()))}
df_feat["region_encoded"] = df_feat["region"].map(region_map)
print(f"  ✓ region_encoded: {region_map}")

# %% [markdown]
# ### 3.4 – Überblick finale Spalten

# %%
new_cols = [
    *hotspot_cols, "dist_km_nearest_hotspot",
    *list(room_dummies.columns),
    "estimated_revenue", "occupancy_rate", "log_price", "region_encoded"
]
print(f"\n  Neue Feature-Spalten ({len(new_cols)}):")
for c in new_cols:
    print(f"    - {c}")

print(f"\n  Finaler Datensatz: {df_feat.shape[0]:,} Zeilen, {df_feat.shape[1]} Spalten")

# %% [markdown]
# ## 4 – Speichern
#
# Wichtig: alle weiteren Notebooks (Modell I – III) sollten ab jetzt
# `cache/20_features_engineered.csv` laden, damit alle drei Modelle
# auf der gleichen Feature-Basis trainiert und fair verglichen werden können.

# %%
print("\n" + "=" * 60)
print("SCHRITT 4: Speichern")
print("=" * 60)

df_feat.to_csv(OUTPUT_CSV, index=False)
print(f"  ✓ Gespeichert: {OUTPUT_CSV}")
print(f"  → {df_feat.shape[0]:,} Zeilen, {df_feat.shape[1]} Spalten")

# %% [markdown]
# ## 5 – Zusammenfassung für das Paper (Abschnitt 2.5)

# %%
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG (für Abschnitt 2.5 im Paper)")
print("=" * 60)
print(f"""
  Datensatz:
    Listings gesamt:   {len(df):,}
    Mit Preis:         {len(df_price):,}
    Nach Bereinigung:  {len(df_clean):,} (Preis ≤ 600 €)

  Neue Features ({len(new_cols)}):
    - 8x Distanz zu Hotspots (km, Haversine)
    - 1x Distanz zum nächsten Hotspot
    - 4x One-Hot room_type
    - 1x estimated_revenue (€/Jahr)
    - 1x occupancy_rate (0–1)
    - 1x log_price
    - 1x region_encoded (0–8)

  Plots gespeichert in: output_plots/
  Feature-CSV:          cache/20_features_engineered.csv
""")

print("=" * 60)
print("  FERTIG! Notebook 30 abgeschlossen.")
print("  Nächster Schritt: notebook_35_random_forest.py")
print("=" * 60)
