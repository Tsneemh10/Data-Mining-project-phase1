import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv('Mall_Customers.csv')
# df['Gender']= df['Gender'].astype('category')
# df['Gender']= df['Gender'].cat.codes
corrMatrix= df.corr()
ax= plt.figure().add_subplot()
sns.heatmap(corrMatrix, cmap='RdBu', vmin=-0.5, vmax=0.5, annot=True)
plt.show()