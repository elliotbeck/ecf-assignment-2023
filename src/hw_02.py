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

# ------------------------------- Load the data ------------------------------ #
data = pd.read_csv('data/wrds_preprocessed.csv')
data['datadate'] = pd.to_datetime(data['datadate'])
# data['fyear'] = pd.to_datetime(data.fyear, format='%Y')
data_ratios = data[['datadate', 'gvkey', 'fyear']]

# ---------------------------------------------------------------------------- #
#                      Calculate the following variables:                      #
# ---------------------------------------------------------------------------- #

# ------------------------------ Book leverage 1 ----------------------------- #
data_ratios['book_leverage_1'] = np.divide(
    (data['dlc'].to_numpy() + data['dltt'].to_numpy()),
    data['at'].to_numpy(),
    where=data['at'] != 0)

# ------------------------------ Book leverage 2 ----------------------------- #
data_ratios['book_leverage_2'] = np.divide(
    data['lt'].to_numpy(),
    data['at'].to_numpy(),
    where=data['at'] != 0)

# ---------------------------- Net book leverage 1 --------------------------- #
data_ratios['net_book_leverage_1'] = np.divide(
    (data['dlc'] + data['dltt'] - data['che']).to_numpy(),
    data['at'].to_numpy(),
    where=data['at'] != 0)

# ----------------------- Common equity at market value ---------------------- #
data_ratios['c_e_at_mv'] = data['csho'].to_numpy() * data['prcc_f'].to_numpy()

# ------------------------------ Market leverage ----------------------------- #
data_ratios['market_leverage'] = np.divide(
    (data['dlc'] + data['dltt']).to_numpy(),
    (data['dlc'] + data['dltt'] + data['pstk'] +
     data_ratios['c_e_at_mv']).to_numpy(),
    where=(data['dlc'] + data['dltt'] + data['pstk'] +
           data_ratios['c_e_at_mv']) != 0)

# ----------------------------- Asset tangibility ---------------------------- #
data_ratios['asset_tangibility'] = np.divide(
    data['ppent'].to_numpy(),
    data['at'].to_numpy(),
    where=data['at'] != 0)

# ------------------- Cash and short-term investments ratio ------------------ #
data_ratios['cash_sti'] = np.divide(
    data['che'].to_numpy(),
    data["at"].to_numpy(),
    where=data["at"] != 0)

# ----------------------------- Return on equity ----------------------------- #
data_ratios['roe'] = np.divide(
    data['niadj'].to_numpy(),
    data['csho'].to_numpy(),
    where=data['csho'] != 0)

# ------------------------------- Profit margin ------------------------------ #
data_ratios['profit_margin'] = np.divide(
    data['niadj'].to_numpy(),
    data['sale'].to_numpy(),
    where=data['sale'] != 0)

# -------------------------------- Capex ratio ------------------------------- #
data_ratios['capex'] = np.divide(
    data['capx'].to_numpy(),
    data['at'].to_numpy(),
    where=data['at'] != 0)

# --------------------------------- R&D ratio -------------------------------- #
data_ratios['rd'] = np.divide(
    data['xrd'].to_numpy(),
    data['at'].to_numpy(),
    where=data['at'] != 0)

# ------------------------------- Dividend yield ------------------------------ #
data_ratios['dividend_yield'] = np.nan
for gvky in data.gvkey.unique():
    data_ratios.loc[data.gvkey == gvky, "dividend_yield"] = np.where(
        (data.loc[data.gvkey == gvky, "fyear"] -
         data.loc[data.gvkey == gvky, "fyear"].shift(1) == 1),
        np.divide(np.divide(data.loc[data.gvkey == gvky, "dv"].to_numpy(),
                            data.loc[data.gvkey == gvky, "csho"].shift(
            1).to_numpy(),
            where=data.loc[data.gvkey == gvky, "csho"].shift(1) != 0),
            data.loc[data.gvkey == gvky, "prcc_f"].shift(
            1).to_numpy(),
            where=data.loc[data.gvkey == gvky, "prcc_f"].shift(1) != 0), np.nan)

# ------------------------------ Dividend payer ------------------------------ #
data_ratios['dividend_payer'] = (data['dv'].to_numpy() > 0).astype(int)

# ---------------------------- Total payout ratio ---------------------------- #
data_ratios['total_payout'] = np.divide(
    (data['dv'] + data['prstkc']).to_numpy(),
    data['niadj'].to_numpy(),
    where=data['niadj'].to_numpy() != 0)

# -------------------------- EBIT interest coverage -------------------------- #
data_ratios['ebit_interest_coverage'] = np.divide(
    (data['oiadp']).to_numpy(),
    data['xint'].to_numpy(),
    where=data['xint'].to_numpy() != 0)

# ---------------------------- Save the ratios df ---------------------------- #
data_ratios.to_csv('data/wrds_ratios.csv', index=False)

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

kpcols = data_ratios.columns[3:]
data_ratios_win = data_ratios.copy()
for year in data_ratios_win.fyear.unique():
    lwr = data_ratios_win.loc[data_ratios_win.fyear ==
                              year, kpcols].quantile(0.01, numeric_only=True)
    ppr = data_ratios_win.loc[data_ratios_win.fyear ==
                              year, kpcols].quantile(0.99, numeric_only=True)
    data_ratios_win.loc[data_ratios_win.fyear == year, kpcols] = data_ratios_win.loc[
        data_ratios_win.fyear == year,
        kpcols].clip(lwr,
                     ppr,
                     axis=1,
                     inplace=False)

# ---------------------------- Save the ratios df ---------------------------- #
data_ratios_win.to_csv('data/wrds_ratios_winsorized.csv', index=False)

# ---------------------------------------------------------------------------- #
#                   b) Create a table with summary statistics                  #
# ---------------------------------------------------------------------------- #

# ----------------------- First, for the whole sample ------------------------ #
data_ratios_summary = data_ratios_win.iloc[:, 3:].describe(
    percentiles=[0.5]).T  # TODO: style.to_latex()
meanmeddiff = data_ratios_summary.iloc[:, 1]-data_ratios_summary.iloc[:, 4]
data_ratios_summary.to_excel(
    'results/2_b_winsorized_complete.xlsx')
data_ratios_win.to_csv('data/wrds_winsorized_ratios_complete.csv',
                       index=False)

# ------------------ Only keep samples above 75th percentile ----------------- #
indices_75_percentile = data.groupby('fyear')['at'].apply(
    lambda x: x[x > x.quantile(0.75)]).drop(
        columns='level_1').index.get_level_values(1)
data_ratios_75_percentile = data_ratios_win.iloc[indices_75_percentile, :]
data_ratios_summary_75_percentile = data_ratios_75_percentile.iloc[:, 3:].describe(
    percentiles=[0.5]).T
meanmeddiff = data_ratios_summary_75_percentile.iloc[:,
                                                     1]-data_ratios_summary_75_percentile.iloc[:, 4]
data_ratios_summary_75_percentile.to_excel(
    'results/2_b_winsorized_75_percentile.xlsx')
data_ratios_75_percentile.to_csv(
    'data/wrds_winsorized_ratios_75_percentile.csv',
    index=False)

# ----------------------------- Compare the means ---------------------------- #
print(data_ratios_summary_75_percentile['mean'] - data_ratios_summary['mean'])

# Reassuringly, the means for common equity at market value are very different


# ---------------------------------------------------------------------------- #
#                           c) Create plots over time                          #
# ---------------------------------------------------------------------------- #
fig, ax = plt.subplots()
data_ratios.groupby(
    'fyear')[['asset_tangibility', 'cash_sti', 'dividend_payer']].mean(numeric_only=True).plot(
    kind='line', ax=ax, xlabel='Year', ylabel='Ratio',
    title='Ratios over time for all companies')
ax.legend(['Asset tangibility', 'Cash and short-term investments',
           'Dividend payer'])
ax.grid()
plt.savefig('results/2_c_ratios_over_time_complete.png')
plt.show()

fig, ax = plt.subplots()
data_ratios_75_percentile.groupby(
    'fyear')[['asset_tangibility', 'cash_sti', 'dividend_payer']].mean().plot(
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
book_market_leverage = data_ratios_75_percentile[[
    'book_leverage_1', 'book_leverage_2', 'net_book_leverage_1',
    'market_leverage']].describe(percentiles=[0.5]).T
