# diabetes-data

#libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_not_zero = ['Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' , 'DiabetesPedigreeFunction' , 'Age']

for c in feature_not_zero :
    data[c] = data[c].replace(0 , np.nan)
    mean = int(data[c].mean(skipna = True))
    data[c] = data[c].replace(np.nan , mean)

norma = StandardScaler()

x = data.iloc[: , :-1]
y = data.iloc[: , -1]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 42)

x_train = norma.fit_transform(x_train)
x_test = norma.transform(x_test)
    
