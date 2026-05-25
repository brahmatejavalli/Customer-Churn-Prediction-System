"""
eda.py  —  Exploratory Data Analysis
-------------------------------------
Run this script to generate all EDA charts
saved as PNGs inside model/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'churn.csv')
df = pd.read_csv(DATA_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {(df['Churn']=='Yes').mean():.1%}\n")

# ── 1. Overall churn distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Customer Churn Overview', fontsize=16, fontweight='bold')

counts = df['Churn'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Churn Distribution')

sns.countplot(data=df, x='Churn', palette=['#4CAF50', '#F44336'], ax=axes[1], hue='Churn', legend=False)
axes[1].set_title('Churn Counts')
axes[1].bar_label(axes[1].containers[0], fontsize=11)
axes[1].bar_label(axes[1].containers[1], fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'eda_01_churn_distribution.png'), dpi=150)
plt.close()

# ── 2. Contract type vs Churn ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Contract Type Analysis', fontsize=16, fontweight='bold')

ct = df.groupby(['Contract', 'Churn']).size().unstack()
ct.plot(kind='bar', ax=axes[0], color=['#4CAF50', '#F44336'], edgecolor='white')
axes[0].set_title('Contract Type vs Churn Count')
axes[0].set_xlabel('Contract Type')
axes[0].set_ylabel('Number of Customers')
axes[0].tick_params(axis='x', rotation=20)

churn_rate_by_contract = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
churn_rate_by_contract.columns = ['Contract', 'Churn_Rate']
sns.barplot(data=churn_rate_by_contract, x='Contract', y='Churn_Rate',
            palette='Reds_d', ax=axes[1], hue='Contract', legend=False)
axes[1].set_title('Churn Rate (%) by Contract Type')
axes[1].set_ylabel('Churn Rate (%)')
for bar in axes[1].patches:
    axes[1].annotate(f'{bar.get_height():.1f}%',
                     (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                     ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'eda_02_contract_vs_churn.png'), dpi=150)
plt.close()

# ── 3. Monthly charges vs Churn ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Charges Analysis vs Churn', fontsize=16, fontweight='bold')

sns.histplot(data=df, x='MonthlyCharges', hue='Churn',
             palette=['#4CAF50', '#F44336'], bins=30, alpha=0.6, ax=axes[0])
axes[0].set_title('Monthly Charges Distribution')

sns.boxplot(data=df, x='Churn', y='MonthlyCharges',
            palette=['#4CAF50', '#F44336'], ax=axes[1], hue='Churn', legend=False)
axes[1].set_title('Monthly Charges by Churn')

sns.boxplot(data=df, x='Churn', y='TotalCharges',
            palette=['#4CAF50', '#F44336'], ax=axes[2], hue='Churn', legend=False)
axes[2].set_title('Total Charges by Churn')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'eda_03_charges_vs_churn.png'), dpi=150)
plt.close()

# ── 4. Tenure vs Churn ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Tenure Analysis', fontsize=16, fontweight='bold')

sns.histplot(data=df, x='tenure', hue='Churn',
             palette=['#4CAF50', '#F44336'], bins=24, alpha=0.6, ax=axes[0])
axes[0].set_title('Tenure Distribution by Churn')
axes[0].set_xlabel('Tenure (months)')

df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 36, 72],
                             labels=['New (0-1yr)', 'Mid (1-3yr)', 'Long (3+yr)'],
                             include_lowest=True)
tenure_churn = df.groupby('tenure_group', observed=True)['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
tenure_churn.columns = ['Tenure Group', 'Churn Rate']
sns.barplot(data=tenure_churn, x='Tenure Group', y='Churn Rate',
            palette='Blues_d', ax=axes[1], hue='Tenure Group', legend=False)
axes[1].set_title('Churn Rate by Tenure Group')
axes[1].set_ylabel('Churn Rate (%)')
for bar in axes[1].patches:
    axes[1].annotate(f'{bar.get_height():.1f}%',
                     (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                     ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'eda_04_tenure_vs_churn.png'), dpi=150)
plt.close()

# ── 5. Internet service & Tech support vs Churn ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Service Features vs Churn', fontsize=16, fontweight='bold')

for ax, col, title in zip(axes,
                           ['InternetService', 'TechSupport'],
                           ['Internet Service', 'Tech Support']):
    rate = df.groupby(col)['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    rate.columns = [col, 'Churn Rate']
    sns.barplot(data=rate, x=col, y='Churn Rate',
                palette='magma', ax=ax, hue=col, legend=False)
    ax.set_title(f'{title} vs Churn Rate')
    ax.set_ylabel('Churn Rate (%)')
    ax.tick_params(axis='x', rotation=15)
    for bar in ax.patches:
        ax.annotate(f'{bar.get_height():.1f}%',
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                    ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'eda_05_services_vs_churn.png'), dpi=150)
plt.close()

print("[INFO] All EDA charts saved to model/ directory")
