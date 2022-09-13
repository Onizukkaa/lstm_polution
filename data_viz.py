"""
UTF-8
Joachim ANDRE
13/09/22

DATA VIZ
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("C:/Users/utilisateur/OneDrive - Simplonformations.co/Bureau/Programmations/LSTM_polution/lstm_polution/LSTM-Multivariate_pollution.csv")

#%%
print(df.head())
# %%
sns.pairplot(df)
# %%
sns.pairplot(df, hue = "pollution")
# %%
df.describe()

# %%
