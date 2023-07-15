import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

#to read the data which is persent in the TSLA.csv file
df = pd.read_csv('TSLA.csv')
#print(df.head()) #this will print only top five values beacuse head is property of panda

#this will print the number of row and collum in our data
# print(df.shape)
(df.shape)

print('\n ')
#to describe each of the data we use this 
print(df.describe())

print('\n ')
print(df.info())

#this is the code to display the closing values of the stock prince with respect to dollor 
# plt.figure(figsize=(12,5))
# plt.plot(df['Close'])
# plt.title('Tesla Close price.', fontsize=15)
# plt.ylabel('Price in dollars.')
# plt.show()

df = df.drop(['Adj Close'], axis=1)
print(df.head()) #this will print only top five values beacuse head is property of panda


#to check any NULL value is present or not
print(df.isnull().sum())


#####this is the code print each of one graph

features = ['Open', 'High', 'Low', 'Close', 'Volume']
# plt.subplots(figsize=(20,10))
# for i, col in enumerate(features):
#     plt.subplot(2,3,i+1)
#     sb.distplot(df[col])

#this is providing the gap ::
# plt.subplots_adjust(hspace=0.5, wspace=0.3)
# plt.show()

#######for checking the outliers in each features: 
# plt.subplots(figsize=(20,10))
# for i, col in enumerate(features):
#     plt.subplot(2,3,i+1)
#     sb.boxplot(df[col])
# plt.show()

#######adding the day , and year part on the code
splitted = df['Date'].str.split('/', expand=True)
df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')
####adding and checking the month is quater or not
df['is_quarter_end'] = np.where(df['month'].isin([3, 6, 9, 12]), 1, 0)
print(df.head())


########### for checking the mean so that we compare the volume of trade######
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format

df['day'] = df['Date'].dt.day #dt.day is an attribute of the pandas DatetimeIndex 
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

data_grouped = df.groupby('year').mean()
print('\n')
print(df.groupby('is_quarter_end').mean())

########again ploting the bar graph to check the mean stock price of each feature
# df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format

# df['day'] = df['Date'].dt.day #dt.day is an attribute of the pandas DatetimeIndex 
# df['month'] = df['Date'].dt.month
# df['year'] = df['Date'].dt.year

# data_grouped = df.groupby('year').mean()
# plt.subplots(figsize=(20, 10))

# for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
#     plt.subplot(2, 2, i+1)
#     data_grouped[col].plot.bar()

# plt.show()

######################checking and printing the quater mean  values#################


############### train our model for better user experiece ##########


