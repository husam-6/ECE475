"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 7

"""

# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# %%
df = pd.read_csv("kaggle_survey_2020_responses.csv")

# %%

# Pick columns we find interesting
q7 = []
for i in range(1, 13):
    q7.append(f"Q7_Part_{i}")

df = df[["Q1", "Q2", "Q3", "Q5", "Q6"] + q7]
df_arr = df.to_numpy()[1:][:]

# Remove NaN values (each row should just be a basket of items...)
dataset = []
for i in range(df_arr.shape[0]):
    truncated = [item for item in filter(lambda v: v==v, df_arr[i])]
    dataset.append(truncated)

# %%

# Transform data into a one hot encoded array 
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
output = pd.DataFrame(te_ary, columns=te.columns_)

# %% Fit apriori model 

apriori(output, min_support=0.3, use_colnames=True)


# %%
df