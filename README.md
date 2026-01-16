# worldbank-refugee-data-analysis
Cleaning, analysis and visualization of World Bank refugee data (1990-2023)

# 🌍 Refugee Population Analysis (1990–2023)

A data cleaning, transformation, and exploratory analysis project using World Bank refugee data.

## 📌 Project Overview
This project processes and analyzes refugee population data from the World Bank (1960–2023), focusing on country-level trends from 1990 onward.

## 🎯 Objectives
- Clean and reshape raw refugee data.
- Remove aggregated and non-country entries.
- Handle missing values appropriately.
- Perform exploratory data analysis (EDA) and visualization.
- Identify key trends in global refugee movements.

## 📂 Dataset
- **Source**: World Bank – Refugee Population by Country or Territory of Asylum
- **Original File**: `Refugee_Dataset.csv`
- **Time Range**: 1960–2023
- **Cleaned Focus**: 1990–2023

## 🛠 Tech Stack
- Python 3.9+
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## 📈 Key Steps
1. **Data Cleaning**
   - Removed unnecessary columns and rows (aggregates, regions, income groups).
   - Renamed columns for clarity.
2. **Missing Value Handling**
   - Dropped sparse early years (1960–1989).
   - Filled later missing values with column means.
3. **Exploratory Analysis**
   - Top refugee-hosting countries.
   - Decadal trends and recent shifts.
4. **Visualization**
   - Bar charts, line graphs, and heatmaps.

## 📊 Sample Insights
- Afghanistan, Syria, and Turkey are among the top refugee-hosting nations.
- Refugee numbers increased significantly after 2010.
- Several countries show minimal refugee populations, indicating stability or data gaps.

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/refugee-data-analysis.git

 #  Install dependencies =   pip install -r requirements.txt
 #Open the Jupyter notebook:  jupyter notebook notebooks/DAS7003_PRAC1.ipynb
