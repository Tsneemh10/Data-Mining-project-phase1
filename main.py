import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Gender'] = df['Gender'].astype('category')
    df['Gender'] = df['Gender'].cat.codes
    return df


def plot_heatmap(df):
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, cmap='RdBu', vmin=-0.5, vmax=0.5, annot=True)


def fit_linear_regression(df):
    x= df.drop(columns= ['Spending Score (1-100)' , 'CustomerID'])
    y= df['Spending Score (1-100)']
    regressor = LinearRegression()
    regressor.fit(x.values, y.values)
    return regressor


def detect_outliers(regressor, df):
    predictedSpending= regressor.predict(df[['Gender','Age', 'Annual Income (k$)']].values)
    actualSpending= df['Spending Score (1-100)']
    residuals=[]
    for i in range(len(predictedSpending)):
        residuals.append(predictedSpending[i]-actualSpending[i])

    residualsSTD= np.std(residuals)

    outliers = []
    for i in range(len(residuals)):
        if (residuals[i]/residualsSTD >=2 or residuals[i]/residualsSTD <=-2 ):
         outliers.append(f"Customer {i} in row no. {i+2} in the data file with spending rate {actualSpending[i]} is an outlier")

    return outliers


def plot_3d(df):
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
    return outliers


def Kmeans_cluster(numerical_dataframe, n_clusters):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    # Extract relevant features for clustering
    X = numerical_dataframe[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Determine optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Perform K-Means clustering with chosen number of clusters (5 in this case)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train)
    y_kmeans = kmeans.predict(X_test)

    # Visualize the clusters in 3D plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_test[y_kmeans == 0, 2], X_test[y_kmeans == 0, 1], X_test[y_kmeans == 0, 3], s=100, c='red', label='Cluster 1')
    ax.scatter(X_test[y_kmeans == 1, 2], X_test[y_kmeans == 1, 1], X_test[y_kmeans == 1, 3], s=100, c='blue', label='Cluster 2')
    ax.scatter(X_test[y_kmeans == 2, 2], X_test[y_kmeans == 2, 1], X_test[y_kmeans == 2, 3], s=100, c='green', label='Cluster 3')
    ax.scatter(X_test[y_kmeans == 3, 2], X_test[y_kmeans == 3, 1], X_test[y_kmeans == 3, 3], s=100, c='cyan', label='Cluster 4')
    ax.scatter(X_test[y_kmeans == 4, 2], X_test[y_kmeans == 4, 1], X_test[y_kmeans == 4, 3], s=100, c='magenta', label='Cluster 5')

    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Age')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('K-Means Clustering')
    ax.legend()
    plt.show()
    return y_kmeans



if __name__ == '__main__':
    file_path = 'mall_customers.csv'
    df = load_data(file_path)

    # Plot heatmap of correlation matrix
    plot_heatmap(df)
    plt.show()

    # Fit linear regression to predict spending score
    regressor = fit_linear_regression(df)

    # Detect outliers
    outliers = detect_outliers(regressor, df)
    print('Outliers:', outliers)

    # Plot 3D visualization of spending score vs income and age
    plot_3d(df)
    plt.show()

    print("The cluster of each point: ", Kmeans_cluster(df, 5))
    Kmeans_cluster(df, 4)
    Kmeans_cluster(df, 3)
    Kmeans_cluster(df, 2)
