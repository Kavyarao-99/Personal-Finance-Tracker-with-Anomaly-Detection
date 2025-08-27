
# ***GENERATING DATASET***


# Commented out IPython magic to ensure Python compatibility.
# %pip install faker

#Generating a synthetic data set

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

#Initializing faker for realistic descriptions

fake = Faker()

#Generating dates, last 6 months

dates = [datetime.now() - timedelta(days=random.randint(1,180)) for _ in range(150)]
dates.sort()

#Categories and typical amounts

categories = {
    "Food" : (5, 50), #Small transactions done frequently
    "Shopping" : (20, 200), #Medium
    "Transportation" : (10, 100),
    "Accommodation" : (800, 1000), #Large (rent/mortgage)
    "Income" : (2000, 5000), #Positive amounts
    "Entertainment" : (10, 100),
    "Miscellaneous" : (1, 1000) #Unexpected transactions (Which could be anamolies)
}

#Generating transactions

transactions = []
for date in dates:
  category = random.choice(list(categories.keys()))
  min_amount, max_amount = categories[category]

  #5% chance of anamoly
  if random.randint(1, 100) <= 5:
    amount = max_amount * random.uniform(2, 10)
  else:
    amount = random.uniform(min_amount, max_amount)

  #Income should be positive and other expenses negative
  if category == "Income":
    amount = abs(amount)
  else:
    amount = -abs(amount)

  transactions.append({
      "Date" : date.strftime("%Y-%m-%d"),
      "Description" : fake.sentence(nb_words = 3)[:-1],
      "Amount" : round(amount, 2),
      "Category" : category
  })

#Create dataframe
df = pd.DataFrame(transactions)

#Adding a few anamolies
df.loc[50, "Amount"] = -5000
df.loc[75, "Amount"] = 10000
df.loc[100, "Amount"] = -0.01 #---> tiny charges, potential fraud

#save to CSV

df.to_csv("Personal_finance.csv", index = False)
print("Dataset Generated with shape: ", df.shape)

#Loading the dataset

import pandas as pd

df = pd.read_csv("Personal_finance.csv")
df.head()

"""# ***DATA CLEANING***"""

#Checking for missing values

print(df.isnull().sum())

#Converting date to datetime

df['Date'] = pd.to_datetime(df['Date'])

#Making sure amount is numeric
df['Amount'] = pd.to_numeric(df['Amount'], errors = 'coerce')

#handling missing values

df = df.dropna(subset= ['Amount', 'Category'])
df['Description'] = df['Description'].fillna('Unknown')

#Removing duplicates

df = df.drop_duplicates(subset = ['Date', 'Description', 'Amount'])

#Checking for typos

valid_categories = ['Food', 'Shopping', 'Transportation', 'Accommodation', 'Income', 'Entertainment', 'Miscellaneous']
df = df[df['Category'].isin(valid_categories)] #----> Keep only valid categories

#Final checking

print("\n", "Cleaned_dataset_summary: ")
print(df.info())
print("\n", "Cleaned_dataset_head: ")
print(df.head())
df.to_csv("personal_finance_clean.csv", index= False)
df.head()

"""# ***ANOMALY*** ***DETECTION***


**Method 1 : Z-Score**
"""

from scipy import stats
import numpy as np

#Calculating Z_scores for 'Amount'
df['Amount_Zscore'] = np.abs(stats.zscore(df['Amount']))

#Flag Anomalies : Zscore > 3
df['Anomaly_Zscore'] = df['Amount_Zscore'] > 3
print('Z-score Anomalies: ', df[df['Anomaly_Zscore']].shape[0])

"""**Method 2 : Isolation Forest**"""

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination = 0.05, random_state = 42) #---> 5% Anomalies
df['Anomaly_ISO'] = model.fit_predict(df[['Amount']]) #----> returns -1 for anomalies
df['Anomaly_ISO'] = df['Anomaly_ISO'].map({1: 0, -1: 1})

print('Isolation forest Anomalies: ', df['Anomaly_ISO'].sum())

"""**Comparison between two methods**"""

both = df[(df['Anomaly_Zscore']) & (df['Anomaly_ISO'] == 1)]
print("Common Anomalies in Both Methods:", both.shape[0])

"""Z-score detected 2 extreme anomalies, while Isolation Forest, being more sensitive to distribution, flagged 8. Two transactions were flagged by both methods, indicating stronger confidence in those points being outliers.

# ***VISUALIZATION***

**1. Distribution of Transaction Amounts**
"""

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))
plt.hist(df['Amount'], bins = 30, color = 'skyblue', edgecolor = 'black')
plt.title('DISTRIBUTION OF TRANSACTIONS')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
plt.savefig('transaction_distribution.png')

"""**2. Spending by Category (Bar Chart)**"""

import matplotlib.pyplot as plt

category_totals = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
category_totals = category_totals.sort_values(ascending = True)

category_totals.sort_values().plot(kind = 'barh', color = 'Teal')
plt.title('TOTAL SPENDING BY CATEGORY')
plt.xlabel('Total Spending ($)')
plt.ylabel('Category')
plt.grid(axis = 'x', linestyle = '--', alpha = 0.7)
plt.savefig('category_spending.png')

"""**3. Amount vs Date (Anomalies highlighted)**"""

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.scatter(df['Date'], df['Amount'],
            c=df['Anomaly_ISO'],
            cmap='coolwarm',
            alpha=0.6,
            s=50)
plt.title('Transactions Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.grid(alpha=0.3)
plt.colorbar(label='Isolation Forest Anomaly (1=Yes)')
plt.savefig('transactions_over_time.png')

"""Insights:
1. Red dots show anomalies detected by Isolation Forest.

1. These unusual transactions are scattered across time, showing no specific monthly pattern.

2. The Isolation Forest algorithm effectively flags outliers across the full range of dates, not just clusters.

**4. Amount_Zscore vs Anomaly_ISO**
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure( figsize = (10, 6))
sns.scatterplot(data = df, x = 'Amount_Zscore', y = 'Amount',
                hue = 'Anomaly_ISO', palette = ['blue', 'red'],
                style = 'Anomaly_ISO', markers = ['o', 'X'], s= 80)
plt.axvline(x=3, color='gray', linestyle='--', alpha=0.7, label='Z-score Threshold')
plt.title('Z-score vs Isolation Forest Anomaly', fontsize = 14)
plt.xlabel('Z-score', fontsize = 12)
plt.ylabel('Amount ($)', fontsize = 12)
plt.legend(title = 'Isolation Forest Anomaly', loc = 'upper right')
plt.grid(axis = 'y', linestyle = '--', alpha = 0.6)
plt.savefig('amount_zscore_vs_anomaly_iso.png')

"""Insights:

1. Most transactions cluster around the z-score between 0 and 1.

2. Some Z-score points are flagged as anomalies (Tiny charges).

3. High Z-scores often correspond to anomalies, but not always.

4. The two methods partially overlap in anomaly detection.

5. Points beyond z=3 are extreme but may not always match ISO anomalies.

**5. Category vs Anomaly_Zscore**
"""

#counting anomalies by category
category_anomalies = df.groupby('Category')['Anomaly_Zscore'].sum().sort_values()

import matplotlib.pyplot as plt

plt.figure(figsize = (10,6))
category_anomalies.plot(kind = 'barh', color = 'purple')
plt.title('Z-score Anomalies by Category', fontsize = 14)
plt.xlabel('Number of Anomalies', fontsize = 12)
plt.ylabel('Category', fontsize = 12)
plt.grid(axis = 'x', linestyle = '--', alpha = 0.6)
plt.savefig('category_anomalies_zscore.png')

"""Insights:

1. Z_score flagged anomalies are exclusively in the Income category.

2. No anomalies detected in spending categories like food, shopping, or transportation.

3. High-income values significantly deviate from the dataset's average transaction behavior.

**6. Amount_Zscore vs Anomaly_Zscore**
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize =(10, 6))
sns.boxplot(data = df,
            x= 'Anomaly_Zscore',
            y= 'Amount_Zscore',
            hue = 'Anomaly_Zscore',
            palette = 'pastel')
plt.title('Z-score Distribution: Normal vs Anomaly', fontsize = 14)
plt.xlabel('Anomaly', fontsize = 12)
plt.ylabel('Z-score', fontsize = 12)
plt.xticks([0,1], ['Normal', 'Anomaly'])
plt.grid(axis = 'y', linestyle = '--', alpha = 0.6)
plt.savefig('amount_zscore_vs_anomaly_zscore.png')

"""Insights:

1. Anomalies have consistently high Z-scores (mostly above 6).

2. Normal values are clustered below a Z-score of 1, with few mild outliers.

3. The Z-score method effectively separates anomalies from the rest of the data.
"""

