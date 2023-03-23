'''
File: hw_04.py
File Created: Friday, 17th March 2023 5:32:58 pm
Author: Elliot Beck (elliot.beck@bf.uzh.ch)
-----
Last Modified: Thursday, 23rd March 2023 8:58:49 am
Modified By: Elliot Beck (elliot.beck@bf.uzh.ch>)
'''

# ---------------------------------------------------------------------------- #
#            Exercise 4: Fiscal Year and calendar year in Compustat            #
# ---------------------------------------------------------------------------- #

# ------------------------------ Load libraries ------------------------------ #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Load in the data ----------------------------- #
data_ratios = pd.read_csv('data/wrds_ratios.csv')
data_wrds = pd.read_csv('data/wrds_preprocessed.csv')

# ------------ Combine data to calculate growth rates per company ------------ #
data_market_value = data_wrds[['fyear', 'fyr', 'gvkey']]
data_market_value = data_market_value.join(data_ratios[['c_e_at_mv']])
data_market_value['c_e_at_mv_yoy'] = data_market_value.groupby(
    'gvkey').c_e_at_mv.pct_change()

# ----------------- Only keep for fiscal years 2000 and 2001 ----------------- #
data_market_value = data_market_value[data_market_value['fyear'].isin([
    2000, 2001])]

# ----------------------------- Remove inf values ---------------------------- #
data_market_value.c_e_at_mv_yoy.replace(
    [np.inf, -np.inf], np.nan, inplace=True)


data_market_value.groupby(
    ['fyear', 'fyr']).c_e_at_mv_yoy.mean().unstack().T
data_market_value.groupby(
    ['fyear', 'fyr']).c_e_at_mv_yoy.std().unstack().T.plot.bar()
plt.show()
