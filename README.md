<<<<<<< HEAD
# Analysing-the-Impact-of-Economic-Crises-on-UK-Living-Standards
A Statistical and Machine Learning Approach
=======
UK Tax Benefits Analysis (1977–2021)
This repository contains a Python-based analysis of the UK tax benefits dataset (1977–2021), focusing on income distribution, inequality, and the redistributive impact of taxes and benefits. The project includes data processing, analytical methods (e.g., Gini coefficient, income growth rates), and visualizations, implemented in a modular codebase. It was developed in Google Colab and is compatible with standard Python environments.
Project Overview
The analysis uses a dataset of mean and median incomes by quintile (1st to 5th) for different income types (Disposable, Final, Gross, Original, Post-tax), in deflated and undeflated forms. Key objectives include:

Quantifying income trends and inequality (e.g., 5th/1st quintile ratio, Gini coefficient).
Assessing the redistributive effects of taxes and benefits.
Exploring relationships with economic indicators (unemployment, inflation).
Visualizing results with line plots, bar plots, and heatmaps.

The codebase is modular, with separate files for data management, cleaning, analysis, visualization, and execution. A comprehensive report (tax_benefits_report.md) summarizes the findings and technical challenges, including the resolution of a TypeError in the redistribution impact plot.
Repository Structure
uk-tax-benefits-analysis/
├── data/
│   └── README.md               # Instructions for obtaining/placing data files
├── data_management.py         # Loads Excel and CSV files
├── data_cleaning.py           # Cleans and transforms data
├── data_analysis.py           # Performs analytical calculations
├── data_visualization.py      # Generates plots
├── main.py                    # Orchestrates the analysis
├── tax_benefits_report.md     # Analysis report
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to ignore
└── LICENSE                    # License terms

Prerequisites

Python: Version 3.8 or higher.
Dependencies: Listed in requirements.txt.
Data Files:
Tax benefits Excel file: tax-benefits-statistics-time-series-v3-filtered-2025-03-17T15-44-252.xlsx
Unemployment CSV: Unemployment_rate_monthly.csv
Inflation CSV: Inflation_monthly.csv
Note: These files are not included due to potential sensitivity. See data/README.md for instructions on obtaining or using placeholder data.



Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/uk-tax-benefits-analysis.git
cd uk-tax-benefits-analysis


Install Dependencies:
pip install -r requirements.txt


Place Data Files:

Copy the required Excel and CSV files to the data/ directory.
Update file paths in main.py if necessary (default: ./data/).
Alternatively, use placeholder data as described in data/README.md.


Run the Analysis:
python main.py


Outputs: Cleaned CSVs (cleaned_tax_benefits_data.csv, cleaned_economic_indicators_data.csv) and 11 PNG plots in the working directory.
Console output includes data previews, statistics, and Granger causality tests.


Google Colab (Optional):

Upload the repository to Google Drive.

Open a Colab notebook and run:
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/uk-tax-benefits-analysis
!pip install -r requirements.txt
%run main.py


Ensure data files are in /content/drive/MyDrive/Data/.




Outputs

Cleaned Data: CSVs with processed tax benefits and economic indicators.
Plots (saved as PNGs):
Disposable income trends, income inequality, Gini coefficient, redistribution impact, etc.
See tax_benefits_report.md for details.


Report: tax_benefits_report.md summarizes findings, including progressive tax-benefit effects and crisis impacts.

Key Findings

The UK tax-benefit system is progressive, with lower quintiles gaining significantly from benefits, especially during the 2008–2009 crisis.
Income inequality peaked in the 1980s and early 2000s but stabilized post-2009.
The Gini coefficient ranged from 0.25 to 0.35, indicating moderate inequality.
Data issues in 2020–2021 (e.g., sharp drops in Final income) suggest COVID-19 impacts or inconsistencies.

Technical Notes

A TypeError in the redistribution impact plot was resolved by enforcing numeric types and handling NaNs (see tax_benefits_report.md).
The codebase is modular and extensible, with debugging outputs for transparency.
Data files are not included; users must provide their own or use placeholders.


