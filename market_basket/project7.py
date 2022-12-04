"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 7

    We chose to run a market basket analysis for the 2020 Kaggle survey results. This survey had almost 20,000 
participants and acquired information such as age range, sex, country of birth, and what programming language you 
use on a regular basis. We felt this dataset could provide key insights into the demographic breakdown of 
programmers. After setting the minimum support to 0.1 and manually screening the results, we found it may be best to 
compare the differences in confidence and lift between the sex's.  As gender bias in the workplace is so prevalent, 
especially in STEM fields involving programming, we wanted to see if women were more likely to learn certain languages 
than others. We were also curious if women would tend to be younger as institutions have recently pushed for women to 
become more involved in STEM fields. Thus, we dropped the support to 0.05 (to account for their being relatively few 
women) and began taking every basket that contained sex and another survey result. A low minimum support made sense 
given the dataset size of nearly 20,000 participants.

    We then compared different results between men and women to see if anything stood out. Our results showcased a few
key differences defined below...

Student:
    Confidence of being a student given man = 0.243
    Lift of being a student given man = 0.943

    Confidence of being a student given women = 0.321
    Lift of being a student given women = 1.243

Having 1-2 Year of Experience: 
    Confidence of having 1-2 years of experience given man = 0.217
    Lift of having 1-2 years of experience given man = 0.967

    Confidence of having 1-2 years of experience given women = 0.258
    Lift of having 1-2 years of experience given women = 1.151

Using Python on a regular basis:
    Confidence of using Python on a regular basis given man = 0.794
    Lift of using Python on a regular basis given man = 1.024

    Confidence of using Python on a regular basis given women = 0.706
    Lift of using Python on a regular basis given women = 0.910

    Our results show that women are more likely to be in school and have only one to two years of experience than men. 
This indicates that recent pushes to get more women interested in programming have been successful.  Further, we found 
that it was more likely for men to use Python than Women. This could be explained by the earlier point that there are 
gender biases heavily prevalent in fields involving programming, and perhaps many men were already using Python in the 
past decade.
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
out = apriori(output, min_support=0.05, use_colnames=True)
out['length'] = out['itemsets'].apply(lambda x: len(x))
doubles = out[out['length'] == 2]
singles = out[out['length'] == 1]

# %% Extract itemsets that have woman or man in it (single frequencies)
man_woman = singles[(singles["itemsets"] == {"Woman"}) | (singles["itemsets"] == {"Man"})]
man_woman = man_woman["support"].to_numpy()

# %%

single_dict = {}
for i, row in singles.iterrows():
    x, = row["itemsets"]
    single_dict[x] = row["support"]

# %% Extract itemsets that have women or man in it (double frequencies)
bool_idx = doubles["itemsets"].apply(lambda x: True if "Man" in x or "Woman" in x else False)
doubles = doubles[bool_idx]

# %%

doubles["confidence"] = doubles.apply(lambda x: x["support"] / man_woman[0] 
                           if "Man" in x["itemsets"] 
                           else x["support"] / man_woman[1], axis = 1) 

doubles.sort_values(by="confidence", ascending=False, inplace=True)

# %%

def parse_itemset(x):
    for item in x:
        if item != "Man" and item != "Woman":
            return item
        

doubles["single"] = doubles["itemsets"].apply(parse_itemset)
doubles["lift"] = doubles.apply(lambda x: x["confidence"] / single_dict[x["single"]], axis=1)

bool_idx = doubles["itemsets"].apply(lambda x: True if "Man" in x else False)
bool_idx_opp = [not x for x in bool_idx]
men = doubles[bool_idx]
women = doubles[bool_idx_opp]

# %%
women

# %%
men