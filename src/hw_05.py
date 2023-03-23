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
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Load in the data ----------------------------- #
data_wrds = pd.read_csv('data/wrds_preprocessed.csv')
data_ratios = pd.read_csv('data/wrds_ratios.csv')
data_ratios['at'] = data_wrds["at"]

# ------------------ Get gykeys for both companies involved ------------------ #
exxon_gykey = data_wrds.loc[data_wrds.conm ==
                            "EXXON MOBIL CORP", :].gvkey.unique()
mobil_gvkey = data_wrds.loc[data_wrds.conm == "MOBIL CORP", :].gvkey.unique()
date_of_merger = data_wrds.loc[data_wrds.conm ==
                               "MOBIL CORP", :].fyear[-1:].values
date_start_analysis = date_of_merger-5
date_end_analysis = date_of_merger+5

# --------------- Create list of book leverage and asset growth -------------- #
data_exxon = data_ratios.loc[data_wrds.gvkey.isin(
    [exxon_gykey[0], mobil_gvkey[0]]), :].copy()

data_exxon_merger_window = data_exxon[data_exxon.fyear.isin(
    list(range(int(date_start_analysis[0]), int(date_end_analysis[0]+1))))]

data_exxon_merger_window.groupby(["gvkey"])[
    ['book_leverage_1', 'book_leverage_2', 'net_book_leverage_1', 'at']].pct_change()
