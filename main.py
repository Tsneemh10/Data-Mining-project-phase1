import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
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
        print("Customer " ,i, " in row no. ", i+2, " in the data file with spending rate ", actualSpending[i] ," is an outlier")

gender = list(df['Gender'])

color = []
for g in gender:
    if g == 'Female':
        color.append('pink')
    else:
        color.append('blue')
X = df[['Annual Income (k$)', 'Age']].values.reshape(-1,2)
Y = df['Spending Score (1-100)']

df2 = pd.DataFrame(X, columns=['Income', 'Age'])
df2['SpendingScore'] = pd.Series(Y)

model = smf.ols(formula='SpendingScore ~ Income + Age', data=df2)
results_formula = model.fit()
results_formula.params

x_surf, y_surf = np.meshgrid(np.linspace(df2.Income.min(), df2.Income.max(), 100),np.linspace(df2.Age.min(), df2.Age.max(), 100))
onlyX = pd.DataFrame({'Income': x_surf.ravel(), 'Age': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)

fittedY=np.array(fittedY)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Income'],df2['Age'],df2['SpendingScore'],c=color)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Income in (k$)')
ax.set_ylabel('Age')
ax.set_zlabel('Spending Score')

pink = mpatches.Patch(color='pink', label='Female')
blue = mpatches.Patch(color='blue', label='Male')
ax.legend(handles=[pink, blue])
outliers = results_formula.outlier_test(method='sidak', alpha=0.05, labels=None, order=False, cutoff= 0.999999)
print( "\nThe outliers using outliers detector are:\n", outliers)
plt.show()