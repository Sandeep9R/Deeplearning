# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:30:42 2020
#Deep learning classification
@author: sravillu
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Data Science\Deep Learning\TensorFlow_FILES\DATA\cancer_classification.csv")

df.columns
sns.distplot(df['benign_0__mal_1'])

# =============================================================================
# check for count of benin and malignant
# =============================================================================

sns.countplot(x='benign_0__mal_1',data=df)

# =============================================================================
# check for null values
# =============================================================================

sns.heatmap(df.isnull()) 

# =============================================================================
# correlation
# =============================================================================



sns.heatmap(df.corr())

# =============================================================================
# determine X and y
# =============================================================================

X=df.iloc[:,:-1].values
y=df['benign_0__mal_1'].values


# =============================================================================
# train test split
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# =============================================================================
# Scale the valriabls
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

# =============================================================================
# Model building 
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model=Sequential()

X_train.shape

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test))


losses=pd.DataFrame(model.history.history)



losses.plot()

#model has opverfitted so lets callback 


# =============================================================================
# Model
# =============================================================================

from tensorflow.keras.callbacks import EarlyStopping

help(EarlyStopping)


early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model=Sequential()

X_train.shape

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])


new_loss=pd.DataFrame(model.history.history)

new_loss.plot()

# =============================================================================
# e\remoival of neurons
# =============================================================================

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5)) #50 pc of neirons are removed 
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5)) #50 pc of neirons are removed 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

finalloss= pd.DataFrame(model.history.history)

finalloss.plot()


# =============================================================================
# predict
# =============================================================================

y_pred=model.predict_classes(X_test)


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))







