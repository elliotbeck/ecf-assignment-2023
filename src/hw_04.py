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
pd.set_option('use_inf_as_na', True)

# ----------------------------- Load in the data ----------------------------- #
data_ratios_win = pd.read_csv('data/wrds_ratios_winsorized.csv')
data_wrds = pd.read_csv('data/wrds_preprocessed.csv')

# ------------ Combine data to calculate growth rates per company ------------ #
data_market_value = data_wrds[['datadate', 'fyear', 'fyr', 'gvkey']]
data_market_value = data_market_value.join(data_ratios_win[['c_e_at_mv']])
data_market_value['c_e_at_mv_yoy'] = data_market_value.groupby(
    'gvkey').c_e_at_mv.pct_change()

# ----------------- Only keep for fiscal years 2000 and 2001 ----------------- #
data_market_value = data_market_value[data_market_value['fyear'].isin([
    2000, 2001])]
data_market_value["fyear"] = data_market_value["fyear"].astype(int)
data_market_value["fyr"] = data_market_value["fyr"].astype(int)

# ------- Calculate the mean and standard deviation of the growth rates ------ #
mean_over_time = data_market_value.groupby(
    ['fyear', 'fyr']).c_e_at_mv_yoy.mean().T
mean_over_time = mean_over_time.reset_index()

sd_over_time = data_market_value.groupby(
    ['fyear', 'fyr']).c_e_at_mv_yoy.std().T
sd_over_time = sd_over_time.reset_index()

# ------------------ Combine, beautify and write to tex file ----------------- #
mean_sd_over_time = pd.concat(
    [mean_over_time, sd_over_time["c_e_at_mv_yoy"]], axis=1)

mean_sd_over_time.columns = ['Fiscal Year', 'Month', 'Mean', 'SD']

mean_sd_over_time = mean_sd_over_time.round(2).astype(
    str).replace(r'\.0$', '', regex=True)

lat_new = mean_sd_over_time.style.set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},],
    overwrite=False)

lat_new = lat_new.hide(axis="index").to_latex(
    column_format='cccc',
    caption='Percentage Change of Common Equity \n by Fiscal year and month')


file_name = "results/mean_sd_over_time_4.tex"  # Include directory path if needed
tex_file = open(file_name, "w")  # This will overwrite an existing file
tex_file.write(lat_new)
tex_file.close()
