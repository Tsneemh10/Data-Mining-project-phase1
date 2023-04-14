import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
df= pd.read_csv('Mall_Customers.csv')
df['Gender']= df['Gender'].astype('category')
df['Gender']= df['Gender'].cat.codes
corrMatrix= df.corr()
ax= plt.figure().add_subplot()
sns.heatmap(corrMatrix, cmap='RdBu', vmin=-0.5, vmax=0.5, annot=True)
x= df.drop(columns= ['Spending Score (1-100)' , 'CustomerID'])
print(x)
y= df['Spending Score (1-100)']
regressor = LinearRegression()
regressor.fit(x.values, y.values)
a = regressor.intercept_
coefficients = regressor.coef_
predictedSpending= regressor.predict(df[['Gender','Age', 'Annual Income (k$)']].values)
actualSpending= y
residuals=[]
for i in range(len(predictedSpending)):
    residuals.append(predictedSpending[i]-actualSpending[i])

residualsSTD= np.std(residuals)

for i in range(len(residuals)):
    if (residuals[i]/residualsSTD >=2 or residuals[i]/residualsSTD <=-2 ):
        print("Customer " ,i+1, " with spending rate ", actualSpending[i] ," is an outlier")

plt.show()