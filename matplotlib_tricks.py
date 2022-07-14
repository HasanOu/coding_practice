import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

df = pd.read_csv('/Users/seyedman/Documents/python_scripts/coding_practice/ISL/An-Introduction-to-Statistical-Learning/data/Auto.csv')

df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = df['horsepower'].astype('float')

string_cols = ' + '.join(df.columns[:-1].difference(['name', 'mpg']))

df['mpg'] = [np.random.randint(0, 2) for _ in range(len(df['mpg']))]

regressors = f'mpg ~ {string_cols} + horsepower*cylinders + np.power(horsepower,2)'
model = smf.logit(regressors, data = df).fit()
print(model.summary())
print(model.resid)
print(model.fittedvalues)
print(model.conf_int)
print(model.params)

# prediction linearRegression

x= np.random.binomial(10, 0.1, 1000)
sns.distplot(x)
plt.savefig('plot.pdf')