'''
File: hw_01.py
File Created: Thursday, 16th March 2023 4:34:27 pm
Author: Elliot Beck (elliot.beck@bf.uzh.ch)
-----
Last Modified: Thursday, 23rd March 2023 8:58:00 am
Modified By: Elliot Beck (elliot.beck@bf.uzh.ch>)
'''

# ---------------------------------------------------------------------------- #
#                           Exercise 1: Preliminaries                          #
# ---------------------------------------------------------------------------- #

# ------------------------------ Load libraries ------------------------------ #
import pandas as pd

# ----------------------------- Read in the data ----------------------------- #
data = pd.read_csv('data/wrds_raw.csv')

# ---------------- Check how many companies are US headquarted --------------- #
print(data['loc'].value_counts())

# -------------------- Only keep US headqurtered companies ------------------- #
num_companies_total = data.gvkey.nunique()
data = data[data['loc'] == 'USA']
num_companies_usa = data.gvkey.nunique()
print(
    f'We have {num_companies_total} companies in total and {num_companies_usa} in the US.')

# ------------------------ Save the preprocessed data ------------------------ #
data.to_csv('data/wrds_preprocessed.csv', index=False)
