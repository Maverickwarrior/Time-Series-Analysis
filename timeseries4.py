#Environment setup
import sklearn
import scipy
import statsmodels
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame
import seaborn as sns
import matplotlib.dates as mdates


df = pd.read_csv('raw_data.csv')
df = pd.read_csv('raw_data.csv', index_col=0, parse_dates=True)
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Weekday Name'] = df.index.weekday_name

df.drop(["asset_id", "asset_name","city_id"], axis = 1, inplace = True)

#Test Harness
split_point = len(df) - 600
dataset, validation = df[0:split_point], df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

X = df.values
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    
    
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse) 


print(df.describe()) 

#Time series analysis using various plots
#1.1 Line plot
df.loc['2019-04-01']
sns.set(rc={'figure.figsize':(11, 4)})
df['count'].plot(linewidth=1.5);

#1.2 Dotted Plot
cols_plot = ['count']
axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
     ax.set_ylabel('counts')
     
     
#1.3 Histogram
df.hist()   

#1.4 Subplot
fig, ax = plt.subplots()
ax.plot(df.loc['2019-01':'2019-01', 'count'], marker='o', linestyle='-')
ax.set_ylabel('counts')
ax.set_title('Jan2019 products sold')


#1.5 Boxplot
sns.boxplot()

#1.6
df['count'].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);


#1.7
#trends
counts = df[['count']]
counts.rolling(12).mean().plot(figsize=(10,10), linewidth=1.5, fontsize=10)
plt.xlabel('Year', fontsize=10);

#1.8 First order differencing
counts.diff().plot(figsize=(10,10), linewidth=1.5, fontsize=10)
plt.xlabel('Year', fontsize=10);

#correlation
df.corr()







