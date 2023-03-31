'''
File: hw_02.py
File Created: Thursday, 16th March 2023 4:20:10 pm
Author: Elliot Beck (elliot.beck@bf.uzh.ch)
-----
Last Modified: Thursday, 23rd March 2023 8:58:13 am
Modified By: Elliot Beck (elliot.beck@bf.uzh.ch>)
'''

# ---------------------------------------------------------------------------- #
#                         Exercise 2: Calculate ratios                         #
# ---------------------------------------------------------------------------- #

# ------------------------------ Load libraries ------------------------------ #
import matplotlib.pyplot as plt
# from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # type: ignore (default='warn')
pd.set_option('use_inf_as_na', True)

# ------------------------------- Load the data ------------------------------ #
data = pd.read_csv('data/wrds_preprocessed.csv')
data['datadate'] = pd.to_datetime(data['datadate'])
# data['fyear'] = pd.to_datetime(data.fyear, format='%Y')
data_ratios = data[['datadate', 'gvkey', 'fyear']]

# ---------------------------------------------------------------------------- #
#                      Calculate the following variables:                      #
# ---------------------------------------------------------------------------- #

# ------------------------------ Book leverage 1 ----------------------------- #
data_ratios['book_leverage_1'] = (data['dlc'] + data['dltt']) / data['at']

# ------------------------------ Book leverage 2 ----------------------------- #
data_ratios['book_leverage_2'] = data['lt'] / data['at']

# ---------------------------- Net book leverage 1 --------------------------- #
data_ratios['net_book_leverage_1'] = (
    (data['dlc'] + data['dltt']) - data['che']) / data['at']

# ----------------------- Common equity at market value ---------------------- #
data_ratios['c_e_at_mv'] = data['csho'] * data['prcc_f']

# ------------------------------ Market leverage ----------------------------- #
data_ratios['market_leverage'] = (data['dlc'] + data['dltt']) / (
    data['dlc'] + data['dltt'] + data['pstk'] + data_ratios['c_e_at_mv'])

# ----------------------------- Asset tangibility ---------------------------- #
data_ratios['asset_tangibility'] = data['ppent'] / data['at']

# ------------------- Cash and short-term investments ratio ------------------ #
data_ratios['cash_sti'] = data['che'] / data["at"]

# ----------------------------- Return on equity ----------------------------- #
data_ratios['roe'] = data['niadj'] / data['teq']

# ------------------------------- Profit margin ------------------------------ #
data_ratios['profit_margin'] = data['niadj'] / data['sale']

# -------------------------------- Capex ratio ------------------------------- #
data_ratios['capex'] = data['capx'] / data['at']

# --------------------------------- R&D ratio -------------------------------- #
data_ratios['rd'] = data['xrd'] / data['at']

# ------------------------------- Dividend yield ------------------------------ #
dividend_yield = data.groupby(['gvkey'])
data_ratios['dividend_yield'] = dividend_yield.apply(
    lambda x: x.assign(dividend_yield=(x.dv/x.csho)/x.prcc_f.shift(1))).dividend_yield

# ------------------------------ Dividend payer ------------------------------ #
dividend_payer = (data.dv > 0).astype(int)
dividend_payer[data.dv.isnull()] = np.NaN
data_ratios['dividend_payer'] = dividend_payer

# ---------------------------- Total payout ratio ---------------------------- #
data_ratios['total_payout'] = (data['dv'] + data['prstkc']) / data['niadj']

# -------------------------- EBIT interest coverage -------------------------- #
data_ratios['ebit_interest_coverage'] = data['oiadp'] / data['xint']

# ---------------------------- Save the ratios df ---------------------------- #
data_ratios.to_csv('data/wrds_ratios.csv', index=False)

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
            "rd": "R\&D Ratio",
            "dividend_payer": "Dividend Payer",
            "dividend_yield": "Dividend Yield",
            "total_payout": "Total Payout Ratio",
            "ebit_interest_coverage": "EBIT Ineterest Coverage"}

# ---------------------------------------------------------------------------- #
#                   a) Winsorize all variables at 1% and 99%                   #
# ---------------------------------------------------------------------------- #
# The winsorize function is not working properly, so I will do it manually
# using a for loop
# for i in data_ratios.columns[3:]:
#     data_ratios[f'{i}_winsorized'] = data_ratios.groupby(
#         'fyear')[i].transform(
#             lambda row: winsorize(
#                 row,
#                 limits=[0.01, 0.01],
#                 nan_policy='omit'))

ratio_cols = data_ratios.columns[3:]
data_ratios_win = data_ratios.copy()
for year in data_ratios_win.fyear.unique():
    lwr = data_ratios_win.loc[data_ratios_win.fyear ==
                              year, ratio_cols].quantile(0.01, numeric_only=True)
    ppr = data_ratios_win.loc[data_ratios_win.fyear ==
                              year, ratio_cols].quantile(0.99, numeric_only=True)
    data_ratios_win.loc[data_ratios_win.fyear == year, ratio_cols] = data_ratios_win.loc[
        data_ratios_win.fyear == year,
        ratio_cols].clip(lwr,
                         ppr,
                         axis=1,
                         inplace=False)

# ---------------------------- Save the ratios df ---------------------------- #
data_ratios_win.to_csv('data/wrds_ratios_winsorized.csv', index=False)
data_ratios_win = data_ratios_win.rename(columns=varnames)

# ---------------------------------------------------------------------------- #
#                   b) Create a table with summary statistics                  #
# ---------------------------------------------------------------------------- #

# ----------------------- First, for the whole sample ------------------------ #
data_ratios_win_describe = data_ratios_win.iloc[:, 3:].describe(
    percentiles=[0.5]).T.round(2)
data_ratios_win_describe["count"] = data_ratios_win_describe["count"].astype(
    int)

# -------------------- Get summary table in correct format ------------------- #
data_ratios_win_describe = data_ratios_win_describe.astype(
    str).replace(r'\.0$', '', regex=True)
data_ratios_win_describe = data_ratios_win_describe.style.set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},],
    overwrite=False).to_latex(
        column_format='lrrrrrr',
        caption='2ba) Summary statistics for winsorized data. (Full sample)')

# ------------- Save the table to a .tex file and to a .csv file ------------- #
file_name = "results/table_2_b_a.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(data_ratios_win_describe)
tex_file.close()

# -------------- Now only for samples above the 75th percentile -------------- #
indices_75_percentile = data.groupby('fyear')['at'].apply(
    lambda x: x[x > x.quantile(0.75)]).drop(
        columns='level_1').index.get_level_values(1)
data_ratios_75_percentile = data_ratios_win.iloc[indices_75_percentile, :]
data_ratios_summary_75_percentile = data_ratios_75_percentile.iloc[:, 3:].describe(
    percentiles=[0.5]).T.round(2)
data_ratios_summary_75_percentile["count"] = data_ratios_summary_75_percentile["count"].astype(
    int)
data_ratios_summary_75_percentile = data_ratios_summary_75_percentile.astype(
    str).replace(r'\.0$', '', regex=True)

data_ratios_summary_75_percentile = data_ratios_summary_75_percentile.style.set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},],
    overwrite=False).to_latex(
        column_format='lrrrrrr',
        caption='2bb) Summary statistics for winsorized data. (Largest 25\% of Companies by Assets)')

# ------------- Save the table to a .tex file and to a .csv file ------------- #
file_name = "results/table_2_b_b.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(data_ratios_summary_75_percentile)
tex_file.close()

data_ratios_75_percentile.to_csv(
    'data/wrds_ratios_winsorized_75_percentile.csv',
    index=False)

# ---------------------------------------------------------------------------- #
#                           c) Create plots over time                          #
# ---------------------------------------------------------------------------- #
fig, ax = plt.subplots()
data_ratios_win.groupby(
    'fyear')[['Asset Tangibility', 'Cash \& Short-term Ratio',
              'Dividend Payer']].mean(numeric_only=True).plot(
    kind='line', ax=ax, xlabel='Year', ylabel='Ratio',
    title='Ratios over time for all companies')
ax.legend(['Asset tangibility', 'Cash and short-term investments',
           'Dividend payer'])
ax.grid()
plt.savefig('results/2_c_ratios_over_time_complete.png')
plt.show()

fig, ax = plt.subplots()
data_ratios_75_percentile.groupby(
    'fyear')[['Asset Tangibility', 'Cash \& Short-term Ratio',
              'Dividend Payer']].mean().plot(
    kind='line', ax=ax, xlabel='Year', ylabel='Ratio',
    title='Ratios over time for large companies')
ax.legend(['Asset tangibility', 'Cash and short-term investments',
           'Dividend payer'])
ax.grid()
plt.savefig('results/2_c_ratios_over_time_75_percentile.png')
plt.show()

# ---------------------------------------------------------------------------- #
#                     d) Book and market leverage variables                    #
# ---------------------------------------------------------------------------- #

# ------------------------------------ a) ------------------------------------ #
data_ratios_75_percentile = data_ratios_win
book_market_leverage = data_ratios_75_percentile[['Book Leverage 1', 'Book Leverage 2', 'Net Book Leverage 1',
                                                  'Market Leverage']]

book_market_leverage = book_market_leverage.describe(
    percentiles=[0.5]).T.round(2)
book_market_leverage["count"] = book_market_leverage["count"].astype(
    int)
book_market_leverage = book_market_leverage.astype(
    str).replace(r'\.0$', '', regex=True)
book_market_leverage = book_market_leverage.style.set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},],
    overwrite=False).to_latex(
        column_format='lrrrrrr',
        caption='2d) Summary statistics of winsorized leverage variables. (Full sample)')

# ------------- Save the table to a .tex file and to a .csv file ------------- #
file_name = "results/table_2_d.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(book_market_leverage)
tex_file.close()
