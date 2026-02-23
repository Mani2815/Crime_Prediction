"""
analysis.py
-----------
Performs exploratory data analysis on the raw crime dataset and saves
static chart images to reports/figures/.  These charts are also served
by the Flask dashboard.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")               # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_PATH  = BASE_DIR / "data" / "raw" / "database.csv"
FIG_DIR    = BASE_DIR / "reports" / "figures"

sns.set_theme(style="whitegrid", palette="muted")


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ── helpers ────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / name}")


# ── individual plots ───────────────────────────────────────────────────────────

def plot_crimes_per_year(df: pd.DataFrame):
    counts = df.groupby("Year").size()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(counts.index, counts.values, marker="o", linewidth=2, color="#2e86ab")
    ax.fill_between(counts.index, counts.values, alpha=0.2, color="#2e86ab")
    ax.set_title("Total Homicide Incidents per Year (1980–2014)", fontsize=14)
    ax.set_xlabel("Year"); ax.set_ylabel("Incident Count")
    _save(fig, "crimes_per_year.png")


def plot_crime_type_distribution(df: pd.DataFrame):
    counts = df["Crime Type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color=["#2e86ab", "#e84855"])
    ax.set_title("Crime Type Distribution", fontsize=14)
    ax.set_xlabel("Crime Type"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "crime_type_dist.png")


def plot_weapon_distribution(df: pd.DataFrame):
    top_weapons = df["Weapon"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_weapons.plot(kind="barh", ax=ax, color="#6a4c93")
    ax.set_title("Top 10 Weapons Used", fontsize=14)
    ax.set_xlabel("Count"); ax.set_ylabel("Weapon")
    ax.invert_yaxis()
    _save(fig, "weapon_dist.png")


def plot_victim_gender(df: pd.DataFrame):
    counts = df["Victim Sex"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    counts.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                colors=["#2e86ab", "#e84855", "#a8dadc"])
    ax.set_title("Victim Gender Distribution", fontsize=14)
    ax.set_ylabel("")
    _save(fig, "victim_gender.png")


def plot_perpetrator_gender(df: pd.DataFrame):
    counts = df["Perpetrator Sex"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    counts.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                colors=["#457b9d", "#e63946", "#a8dadc"])
    ax.set_title("Perpetrator Gender Distribution", fontsize=14)
    ax.set_ylabel("")
    _save(fig, "perpetrator_gender.png")


def plot_age_distribution(df: pd.DataFrame):
    # Perpetrator Age may be object; coerce
    ages = pd.to_numeric(df["Perpetrator Age"], errors="coerce")
    ages = ages[(ages > 0) & (ages < 100)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ages, bins=40, color="#f4a261", edgecolor="white")
    ax.set_title("Perpetrator Age Distribution", fontsize=14)
    ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
    _save(fig, "perp_age_dist.png")


def plot_victim_age_distribution(df: pd.DataFrame):
    ages = pd.to_numeric(df["Victim Age"], errors="coerce")
    ages = ages[(ages >= 0) & (ages < 100)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ages, bins=40, color="#2a9d8f", edgecolor="white")
    ax.set_title("Victim Age Distribution", fontsize=14)
    ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
    _save(fig, "victim_age_dist.png")


def plot_monthly_trend(df: pd.DataFrame):
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    counts = df["Month"].value_counts().reindex(month_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="#264653")
    ax.set_title("Homicides by Month", fontsize=14)
    ax.set_xlabel("Month"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "monthly_trend.png")


def plot_top_states(df: pd.DataFrame):
    top = df["State"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", ax=ax, color="#e9c46a")
    ax.set_title("Top 15 States by Homicide Count", fontsize=14)
    ax.set_xlabel("Count"); ax.set_ylabel("State")
    ax.invert_yaxis()
    _save(fig, "top_states.png")


def plot_relationship_distribution(df: pd.DataFrame):
    top = df["Relationship"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(11, 5))
    top.plot(kind="bar", ax=ax, color="#e76f51")
    ax.set_title("Top 12 Victim–Perpetrator Relationships", fontsize=14)
    ax.set_xlabel("Relationship"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "relationship_dist.png")


def plot_crime_solved(df: pd.DataFrame):
    counts = df["Crime Solved"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    counts.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                colors=["#06d6a0", "#ef476f"])
    ax.set_title("Crime Solved vs Unsolved", fontsize=14)
    ax.set_ylabel("")
    _save(fig, "crime_solved.png")


def plot_heatmap_month_year(df: pd.DataFrame):
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    pivot = df.pivot_table(
        index="Month", columns="Year", values="Incident",
        aggfunc="count", fill_value=0
    )
    pivot = pivot.reindex(month_order)
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Homicide Count: Month × Year", fontsize=14)
    _save(fig, "heatmap_month_year.png")


# ── main ───────────────────────────────────────────────────────────────────────

def run_analysis(data_path=DATA_PATH):
    print("[analysis] Running EDA …")
    df = load_raw(data_path)

    plot_crimes_per_year(df)
    plot_crime_type_distribution(df)
    plot_weapon_distribution(df)
    plot_victim_gender(df)
    plot_perpetrator_gender(df)
    plot_age_distribution(df)
    plot_victim_age_distribution(df)
    plot_monthly_trend(df)
    plot_top_states(df)
    plot_relationship_distribution(df)
    plot_crime_solved(df)
    plot_heatmap_month_year(df)

    print(f"[analysis] All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run_analysis()
