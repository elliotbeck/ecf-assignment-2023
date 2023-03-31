'''
File: hw_05.py
File Created: Thursday, 23rd March 2023 8:23:16 am
Author: Elliot Beck (elliot.beck@bf.uzh.ch)
-----
Last Modified: Thursday, 23rd March 2023 8:57:33 am
Modified By: Elliot Beck (elliot.beck@bf.uzh.ch>)
'''

# ---------------------------------------------------------------------------- #
#                     Mergers and Acquisitions in Compustat                    #
# ---------------------------------------------------------------------------- #

# ------------------------------ Load libraries ------------------------------ #
import pandas as pd
pd.set_option('use_inf_as_na', True)

# ----------------------------- Load in the data ----------------------------- #
data_wrds = pd.read_csv('data/wrds_preprocessed.csv')
data_ratios = pd.read_csv('data/wrds_ratios.csv')
data_ratios['at'] = data_wrds["at"]

data_wrds.conm[data_wrds.conm.str.contains('time warner', case=False)]

# ------------------ Get gykeys for both companies involved ------------------ #
exxon_gykey = data_wrds.loc[data_wrds.conm ==
                            "EXXON MOBIL CORP", :].gvkey.unique()
mobil_gvkey = data_wrds.loc[data_wrds.conm == "MOBIL CORP", :].gvkey.unique()
date_of_merger = data_wrds.loc[data_wrds.conm ==
                               "MOBIL CORP", :].fyear[-1:].values
date_start_analysis = date_of_merger-5
date_end_analysis = date_of_merger+6

# --------------- Create list of book leverage and asset growth -------------- #
data_exxon = data_ratios.loc[data_wrds.gvkey.isin(
    [exxon_gykey[0], mobil_gvkey[0]]), :].copy()

data_exxon_merger_window = data_exxon[data_exxon.fyear.isin(
    list(range(int(date_start_analysis[0]), int(date_end_analysis[0]+1))))]


data_exxon = data_wrds.loc[data_wrds.gvkey.isin(
    [exxon_gykey[0], mobil_gvkey[0]]), :].copy()

data_exxon = data_exxon[data_exxon.fyear.isin(
    list(range(int(date_start_analysis[0]), int(date_end_analysis[0]+1))))]
data_exxon[['fyear', 'dlc', 'dltt', 'lt', 'che', 'at']]

data_exxon = data_exxon.groupby('datadate')[[
    'dlc', 'dltt', 'lt', 'che', 'at']].apply(lambda x: x.sum())

data_exxon['book_leverage_1'] = (
    (data_exxon['dlc'] + data_exxon['dltt']) / data_exxon['at'])
data_exxon['book_leverage_2'] = (
    data_exxon['lt'] / data_exxon['at'])
data_exxon['net_book_leverage_1'] = ((
    (data_exxon['dlc'] + data_exxon['dltt']) - data_exxon['che']) / data_exxon['at'])

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
            "ebit_interest_coverage": "EBIT Ineterest Coverage",
            "datadate": "Date",
            "at": "Total assets"}

# ------------------------ Beautify and write to latex ----------------------- #
exxon_merger = data_exxon[['book_leverage_1',
                           'book_leverage_2', 'net_book_leverage_1', 'at']]

exxon_merger_growth_rates = exxon_merger.pct_change()
exxon_merger_growth_rates.reset_index(inplace=True)
exxon_merger_growth_rates.dropna(inplace=True)
exxon_merger_growth_rates.rename(columns=varnames, inplace=True)
exxon_merger_growth_rates.dropna(inplace=True)
exxon_merger_growth_rates.round(2).astype(str).replace(r'\.0$', '', regex=True)
exxon_merger_growth_rates.reset_index(drop=True, inplace=True)
exxon_merger_growth_rates.to_latex(
    "results/exxon_merger_growth_rates_5.tex",
    index=False,
    caption="Exxon Mobil Merger Book Leverage and Asset Growth")
