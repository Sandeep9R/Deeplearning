# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:13:29 2020
#deep learning regression 
@author: sravillu
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Data Science\Deep Learning\TensorFlow_FILES\DATA\kc_house_data.csv")


sns.distplot(df['price'])

df.isnull()
sns.heatmap(df.isnull())

df.describe().transpose()


sns.countplot(df['bedrooms'])


df.corr()

sns.heatmap(df.corr())

df.corr()['price'].sort_values()

# here we see that correlation between sq_living is highjest


plt.Figure(figsize=(10,5))
sns.scatterplot(x='price',y='sqft_living',data=df)


plt.Figure(figsize=(10,5))


sns.scatterplot(x='price',y='long',data=df)

sns.scatterplot(x='price',y='lat',data=df)


plt.Figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',data=df)

# =============================================================================
# add hue to color in terms of price
# =============================================================================

sns.scatterplot(x='long',y='lat',hue='price',data=df)


# =============================================================================
# removal of outlliers
# =============================================================================

new_df=df.sort_values('price',ascending=False)


# =============================================================================
# remove the top 1 percent
# =============================================================================


len(df)


new_df=df.sort_values('price',ascending=False).iloc[216:,:] #removes 216 or 1 pc 



# =============================================================================
# verify using plot 
# =============================================================================


sns.scatterplot(x='long',y='lat',data=new_df,alpha=0.2,palette='RdYlGn',hue='price')

#change color 


# =============================================================================
# dropping columns 
# =============================================================================

new_df=new_df.drop('id',axis=1)
new_df=new_df.drop('zipcode',axis=1)


# =============================================================================
# convert date to months and year
# =============================================================================

new_df['date']=pd.to_datetime(new_df['date'])

new_df['year']=new_df['date'].apply(lambda date: date.year)

new_df['month']=new_df['date'].apply(lambda date: date.month)


new_df=new_df.drop('date',axis=1)

# =============================================================================
# preprocessing 
# =============================================================================

X=new_df.drop('price',axis=1).values

y=new_df['price'].values


# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# =============================================================================
# scale the data
# =============================================================================

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


# =============================================================================
# Model building
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train.shape

#19 columns 
model=Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)

model.history.history

lossdf=pd.DataFrame(model.history.history)



lossdf.plot()

y_pred=model.predict(X_test)
# =============================================================================
# metrics
# =============================================================================
 

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

error1=mean_squared_error(y_test,y_pred)

rootmse=np.sqrt(mean_squared_error(y_test,y_pred))

v=explained_variance_score(y_test,y_pred)


# =============================================================================
# predict on future prices
# =============================================================================

new_house=new_df.drop('price',axis=1).iloc[0]

# make it 2d array

new_house.values.reshape(-1,19)

#transform

new_house=scaler.transform(new_house.values.reshape(-1,19))


model.predict(new_house)


new_df.head(1)












