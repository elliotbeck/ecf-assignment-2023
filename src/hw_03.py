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
from statsmodels.formula.api import ols
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer

# ----------------------------- Load in the data ----------------------------- #
data_wrds = pd.read_csv('data/wrds_preprocessed.csv')
data_ratios = pd.read_csv('data/wrds_ratios.csv')
data_ratios_win = pd.read_csv('data/wrds_ratios_winsorized.csv')

# ---------------------------------------------------------------------------- #
#                    a) Pooled OLS with non-winsorized data                    #
# ---------------------------------------------------------------------------- #

# ------------------------ Prepare data for pooled ols ----------------------- #
data_a = pd.DataFrame(data_ratios['book_leverage_1']).join(
    np.log(data_wrds['at'], where=(data_wrds['at'].to_numpy() > 0)))
data_a = data_a.join(data_ratios[['asset_tangibility',
                                  'rd',
                                  'dividend_payer',
                                  'profit_margin',
                                  'fyear']])

# ------------------------------- Fit the model ------------------------------ #
reg_a = ols(formula="book_leverage_1 ~ at+asset_tangibility+rd+dividend_payer+profit_margin",
            data=data_a).fit()

# ---------------------------------------------------------------------------- #
#                      b) Pooled OLS with winsorized data                      #
# ---------------------------------------------------------------------------- #
data_b = pd.DataFrame(data_ratios_win['book_leverage_1']).join(
    np.log(data_wrds['at'], where=(data_wrds['at'].to_numpy() > 0)))
data_b = data_b.join(data_ratios_win[['asset_tangibility',
                                      'rd',
                                      'dividend_payer',
                                      'profit_margin',
                                      'fyear']])

# ------------------------------- Fit the model ------------------------------ #
reg_b = ols(formula="book_leverage_1 ~ at+asset_tangibility+rd+dividend_payer+profit_margin",
            data=data_b).fit()

# ---------------------------------------------------------------------------- #
#                     c) Restriced book leverage pooled OLS                    #
# ---------------------------------------------------------------------------- #
data_c = data_b.copy()
data_c['book_leverage_1'] = data_c['book_leverage_1'].clip(0, 1)

# ------------------------------- Fit the model ------------------------------ #
reg_c = ols(formula="book_leverage_1 ~ at+asset_tangibility+rd+dividend_payer+profit_margin",
            data=data_c).fit()

# --------------------------- Compare a), b) and c) -------------------------- #
tab = Stargazer([reg_a, reg_b, reg_c])
tab.custom_columns(['Original', 'Winsorized', "Restricted"], [1, 1, 1])
tab.show_model_numbers(False)
tab.render_latex()

# ---------------------------------------------------------------------------- #
#                               d) Fixed effects                               #
# ---------------------------------------------------------------------------- #

# ---------------------------- Time fixed effects ---------------------------- #
data_d = pd.DataFrame(data_ratios_win['book_leverage_1']).join(
    np.log(data_wrds['at'], where=(data_wrds['at'].to_numpy() > 0)))
data_d = data_d.join(data_ratios_win[['asset_tangibility',
                                      'rd',
                                      'dividend_payer',
                                      'profit_margin',
                                      'fyear']])

data_d["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
# data_d["sic"] = pd.Categorical(data_d.sic)
# data_d["fyear"] = pd.Categorical(data_ratios_win.fyear)
data_d["fyear_sic"] = pd.Categorical(
    data_d.fyear.astype("string") + data_d.sic.astype("string"))
data_d.set_index(['sic', 'fyear'], inplace=True)
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

# ---------------------------------------------------------------------------- #
#      e) Panel OLS with time and industry fixed effects unwisorized data      #
# ---------------------------------------------------------------------------- #
data_e = pd.DataFrame(data_ratios['book_leverage_1']).join(
    np.log(data_wrds['at'], where=(data_wrds['at'].to_numpy() > 0)))
data_e = data_e.join(data_ratios[['asset_tangibility',
                                  'rd',
                                  'dividend_payer',
                                  'profit_margin',
                                  'fyear']])

data_e["sic"] = data_wrds.sic.astype("string").str[:2].astype(int)
data_e["fyear_sic"] = data_e.fyear.astype(
    "string") + data_e.sic.astype("string")

data_e.set_index(['sic', 'fyear'], inplace=True)

# -------------------- Extract the data from the DataFrame ------------------- #
exog = data_e.iloc[:, 1:-1]
exog = sm.add_constant(exog)
target = data_e.iloc[:, 0]

# ------------------------------- Fit the model ------------------------------ #
reg_e = PanelOLS(target,
                 exog,
                 time_effects=True,
                 entity_effects=True)

fit_e = reg_e.fit(
    cov_type="clustered",
    cluster_entity=True,
    cluster_time=True)

reg_c.summary()
fit_e
