'''
File: hw_03.py
File Created: Friday, 17th March 2023 1:20:20 pm
Author: Elliot Beck (elliot.beck@bf.uzh.ch)
-----
Last Modified: Thursday, 23rd March 2023 8:58:38 am
Modified By: Elliot Beck (elliot.beck@bf.uzh.ch>)
'''

# ---------------------------------------------------------------------------- #
#       Exercise 3: The importance of outliers and time-series dependence      #
# ---------------------------------------------------------------------------- #

# ------------------------------ Load libraries ------------------------------ #
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.panel.results import compare

# ----------------------------- Load in the data ----------------------------- #
data_wrds = pd.read_csv('data/wrds_preprocessed.csv')
data_ratios = pd.read_csv('data/wrds_ratios.csv')
data_ratios_win = pd.read_csv('data/wrds_ratios_winsorized.csv')

# ---------------------------------------------------------------------------- #
#                       Get mapping of keys to variables                       #
# ---------------------------------------------------------------------------- #
varnames = {"book_leverage_1": "Book Leverage 1",
            "book_leverage_2": "Book Leverage 2",
            "net_book_leverage_1": "Net Book Leverage 1",
            "c_e_at_mv": "Common Equity",
            "market_leverage": "Market Leverage",
            "asset_tangibility": "Asset Tangibility",
            "cash_sti": "Cash \& Short-term Ratio",
            "roe": "Return on Equity",
            "profit_margin": "Profit Margin",
            "capex": "CapEx Ratio",
            "rd": "R&D Ratio",
            "dividend_payer": "Dividend Payer",
            "dividend_yield": "Dividend Yield",
            "total_payout": "Total Payout Ratio",
            "ebit_interest_coverage": "EBIT Ineterest Coverage",
            "at": "Total Assets",
            "fyear": "Fiscal Year"}

data_wrds = data_wrds.rename(columns=varnames)
data_ratios = data_ratios.rename(columns=varnames)
data_ratios_win = data_ratios_win.rename(columns=varnames)

# ---------------------------------------------------------------------------- #
#                    a) Pooled OLS with non-winsorized data                    #
# ---------------------------------------------------------------------------- #

# ------------------------ Prepare data for pooled ols ----------------------- #
data_a = pd.DataFrame(data_ratios['Book Leverage 1'])
total_assets = np.log(data_wrds['Total Assets'],
                      where=(data_wrds['Total Assets'] > 0))
total_assets[data_wrds['Total Assets'].isnull()] = np.NaN
data_a["Total Assets"] = total_assets
data_a = data_a.join(data_ratios[['Asset Tangibility',
                                  'R&D Ratio',
                                  'Dividend Payer',
                                  'Profit Margin',
                                  'Fiscal Year']])


data_a["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
data_a["fyear_sic"] = data_a["Fiscal Year"].astype(
    "string") + data_a.sic.astype("string")

data_a['fyear'] = data_a['Fiscal Year']
data_a.set_index(['sic', 'fyear'], inplace=True)
data_a.replace([np.inf, -np.inf], np.nan, inplace=True)
data_a.dropna(inplace=True)

# -------------------- Extract the data from the DataFrame ------------------- #
exog = data_a.iloc[:, 1:-2]
exog = sm.add_constant(exog)
target = data_a.iloc[:, 0]


# ------------------------------- Fit the model ------------------------------ #
reg_a = PooledOLS(target,
                  exog).fit()

reg_a.summary

# ---------------------------------------------------------------------------- #
#                      b) Pooled OLS with winsorized data                      #
# ---------------------------------------------------------------------------- #
data_b = pd.DataFrame(data_ratios_win['Book Leverage 1'])
total_assets = np.log(data_wrds['Total Assets'],
                      where=(data_wrds['Total Assets'] > 0))
total_assets[data_wrds['Total Assets'].isnull()] = np.NaN
data_b["Total Assets"] = total_assets
data_b = data_b.join(data_ratios_win[['Asset Tangibility',
                                      'R&D Ratio',
                                      'Dividend Payer',
                                      'Profit Margin',
                                      'Fiscal Year']])


data_b["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
data_b["fyear_sic"] = data_b["Fiscal Year"].astype(
    "string") + data_b.sic.astype("string")

data_b['fyear'] = data_b['Fiscal Year']
data_b.set_index(['sic', 'fyear'], inplace=True)
data_b.replace([np.inf, -np.inf], np.nan, inplace=True)
data_b.dropna(inplace=True)

# -------------------- Extract the data from the DataFrame ------------------- #
exog = data_b.iloc[:, 1:-2]
exog = sm.add_constant(exog)
target = data_b.iloc[:, 0]

# ------------------------------- Fit the model ------------------------------ #
reg_b = PooledOLS(target,
                  exog).fit()

# ---------------------------------------------------------------------------- #
#                     c) Restriced book leverage pooled OLS                    #
# ---------------------------------------------------------------------------- #
data_c = data_b.copy()
data_c['Book Leverage 1'] = data_c['Book Leverage 1'].clip(0, 1)

# ------------------------------- Fit the model ------------------------------ #
reg_c = PooledOLS(data_c['Book Leverage 1'],
                  exog).fit()

# --------------------------- Compare a), b) and c) -------------------------- #
table = {
    '(1)': reg_a,
    '(2)': reg_b,
    '(3)': reg_c,
}

comparison = compare(table)
summary = comparison.summary

file_name = "results/reg_3_a_b_c.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(summary.as_latex())
tex_file.close()

# ---------------------------------------------------------------------------- #
#                               d) Fixed effects                               #
# ---------------------------------------------------------------------------- #

# ---------------------------- Time fixed effects ---------------------------- #
data_d = pd.DataFrame(data_ratios_win['Book Leverage 1'])
total_assets = np.log(data_wrds['Total Assets'],
                      where=(data_wrds['Total Assets'] > 0))
total_assets[data_wrds['Total Assets'].isnull()] = np.NaN
data_d["Total Assets"] = total_assets
data_d = data_d.join(data_ratios_win[['Asset Tangibility',
                                      'R&D Ratio',
                                     'Dividend Payer',
                                      'Profit Margin',
                                      'Fiscal Year']])

data_d["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
data_d["fyear_sic"] = pd.Categorical(
    data_d['Fiscal Year'].astype("string") + data_d.sic.astype("string"))
data_d.set_index(['sic', 'Fiscal Year'], inplace=True)
data_d.replace([np.inf, -np.inf], np.nan, inplace=True)
data_d.dropna(inplace=True)

# -------------------- Extract the data from the DataFrame ------------------- #
exog = data_d.iloc[:, 1:-1]
exog = sm.add_constant(exog)
target = data_d.iloc[:, 0]

# ------------------------------- Fit the models ----------------------------- #
reg_d_1 = PanelOLS(target,
                   exog,
                   time_effects=True)

fit_d_1 = reg_d_1.fit()

reg_d_2 = PanelOLS(target,
                   exog,
                   entity_effects=True,
                   time_effects=True)

fit_d_2 = reg_d_2.fit()

reg_d_3 = PanelOLS(target,
                   exog,
                   other_effects=data_d.fyear_sic)

fit_d_3 = reg_d_3.fit()


# ---------------------------- Write the tex files --------------------------- #
table = {
    '(1)': fit_d_1,
    '(2)': fit_d_2,
    '(3)': fit_d_3,
}

comparison = compare(table)
summary = comparison.summary

file_name = "results/reg_3_d.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(summary.as_latex())
tex_file.close()

# ---------------------------------------------------------------------------- #
#      e) Panel OLS with time and industry fixed effects unwisorized data      #
# ---------------------------------------------------------------------------- #
data_e = pd.DataFrame(data_ratios['Book Leverage 1'])
total_assets = np.log(data_wrds['Total Assets'],
                      where=(data_wrds['Total Assets'] > 0))
total_assets[data_wrds['Total Assets'].isnull()] = np.NaN
data_e["Total Assets"] = total_assets
data_e = data_e.join(data_ratios[['Asset Tangibility',
                                  'R&D Ratio',
                                 'Dividend Payer',
                                  'Profit Margin',
                                  'Fiscal Year']])

data_e["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
data_e["fyear_sic"] = data_e["Fiscal Year"].astype(
    "string") + data_e.sic.astype("string")
data_e["gvkey"] = data_wrds["gvkey"]

data_e.set_index(['sic', 'Fiscal Year'], inplace=True)
data_e.replace([np.inf, -np.inf], np.nan, inplace=True)
data_e.dropna(inplace=True)

# -------------------- Extract the data from the DataFrame ------------------- #
exog = data_e.iloc[:, 1:-2]
exog = sm.add_constant(exog)
target = data_e.iloc[:, 0]

# ------------------------------- Fit the model ------------------------------ #
reg_e_1 = PanelOLS(target,
                   exog,
                   time_effects=True,
                   entity_effects=True)

fit_e_1 = reg_e_1.fit(cov_type="clustered", clusters=data_e["gvkey"])
fit_e_1.summary

# ------------------------------- Again as in c ------------------------------ #
data_e = pd.DataFrame(data_ratios_win['Book Leverage 1'])
total_assets = np.log(data_wrds['Total Assets'],
                      where=(data_wrds['Total Assets'] > 0))
total_assets[data_wrds['Total Assets'].isnull()] = np.NaN
data_e["Total Assets"] = total_assets
data_e = data_e.join(data_ratios_win[['Asset Tangibility',
                                      'R&D Ratio',
                                     'Dividend Payer',
                                      'Profit Margin',
                                      'Fiscal Year']])
data_e["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
data_e["fyear_sic"] = data_e["Fiscal Year"].astype(
    "string") + data_e.sic.astype("string")

data_e.set_index(['sic', 'Fiscal Year'], inplace=True)
data_e.replace([np.inf, -np.inf], np.nan, inplace=True)
data_e.dropna(inplace=True)
data_e['Book Leverage 1'] = data_e['Book Leverage 1'].clip(0, 1)

# -------------------- Extract the data from the DataFrame ------------------- #
exog = data_e.iloc[:, 1:-1]
exog = sm.add_constant(exog)
target = data_e.iloc[:, 0]

# ------------------------------- Fit the model ------------------------------ #
reg_e_2 = PanelOLS(target,
                   exog,
                   time_effects=False,
                   entity_effects=False)

fit_e_2 = reg_e_2.fit()


# ---------------------------- Write the tex files --------------------------- #
table = {
    '(1)': fit_e_1,
    '(2)': fit_e_2
}

comparison = compare(table)
summary = comparison.summary

file_name = "results/reg_3_e.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(summary.as_latex())
tex_file.close()
