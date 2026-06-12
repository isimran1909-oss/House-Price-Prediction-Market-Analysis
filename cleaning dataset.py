import pandas as pd
import numpy as np

df=pd.read_excel('c:/Users/dell/OneDrive/Documents/Data science projects/HousePrediction/House_Price_Dirty_Dataset.xlsx')
house=df.copy()

# print(house.info())
# cleaning location
house['Location']=house['Location'].str.strip().astype('str')
house.loc[(house['Location']=='mumbai','Location')]='Mumbai'
house.loc[(house['Location']=='MUMBAI','Location')]='Mumbai'
house.loc[(house['Location']=='Bangalore','Location')]='Banglore'
house.loc[(house['Location']=='bangalore','Location')]='Banglore'
house.loc[(house['Location']=='Delhi','Location')]='Delhi'

# cleaning area
house['Area_sqft']=house['Area_sqft'].astype(str)
house['Area_sqft']=house['Area_sqft'].str.replace(' sqft','',regex=False)
house['Area_sqft']=pd.to_numeric(house['Area_sqft'],errors='coerce')
house['Area_sqft']=house['Area_sqft'].fillna(house['Area_sqft'].median())


# cleaning price

house['Price']=house['Price'].fillna(house['Price'].median())
# print(house['Price'].isnull().sum())


# house.to_csv('c:/Users/dell/OneDrive/Documents/Data science projects/HousePrediction/House_Price_cleaned.csv')

# MACHINE LEARNING

from sklearn.model_selection import train_test_split


# removing outliers from price

Q1 = house['Price'].quantile(0.25)
Q3 = house['Price'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# print(lower)
# print(upper)
# print(house[house['Price'] > 10191395].shape)# identifying the outliers
house[house['Price'] > 10191395]['Price'].sort_values()
house_clean = house[
    house['Price'] <= 10191395
]
# print("Before:", house.shape)
# print("After :", house_clean.shape)
house_clean.corr(numeric_only=True)['Price'].sort_values(ascending=False)

X=house_clean[['Location','Area_sqft','Bedrooms','Bathrooms','House_Age','Parking','Furnishing','Distance_City_km']]
y=house_clean['Price']

X=pd.get_dummies(X,columns=['Location','Furnishing'],drop_first=True)

# spliting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

corr_with_price = house_clean.corr(numeric_only=True)['Price']
# print(corr_with_price)

# print(house_clean.isnull().sum())


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

# predict
predictions=lr.predict(X_test)

result=pd.DataFrame({'Actual value':y_test,
                     'predicted value':predictions})

result['Actual value'] = result['Actual value'].map('{:,.0f}'.format)
result['predicted value'] = result['predicted value'].map('{:,.0f}'.format)

# print(result.head(10))
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
r2=r2_score(y_test,predictions)
MSE=mean_squared_error(y_test,predictions)
MAE=mean_absolute_error(y_test,predictions)

metrices = pd.DataFrame({
    'Metric': ['R² Score', 'MAE', 'MSE'],
    'Value': [r2, MAE, MSE]
})

# metrices.to_csv('c:/Users/dell/OneDrive/Documents/Data science projects/HousePrediction/metrices.csv', index=False)
# print(accuracyy)
# house_clean.to_csv("c:/Users/dell/OneDrive/Documents/Data science projects/HousePrediction/House_cleanedOutliers.csv")
# result.to_csv('c:/Users/dell/OneDrive/Documents/Data science projects/HousePrediction/result.csv')

