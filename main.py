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

def elbow_best_number_of_clusters(df):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn_extra.cluster import KMedoids
    # Determine optimal number of clusters using the elbow method
    # Extract relevant features for clustering
    X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    wcss_kmeans = []
    k_range = range(1, 11)
    cost = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss_kmeans.append(kmeans.inertia_)

        # Calculate sum of dissimilarities for each value of k
        kmedoids = KMedoids(n_clusters=i, random_state=0)
        kmedoids.fit(X)
        cost.append(kmedoids.inertia_)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax.plot(k_range, cost)
    ax.set_title('sum of dissimilarities for kmedoids')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Cost')

    ax2.plot(k_range, wcss_kmeans)
    ax2.set_title('Elbow Method for kmeans')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('WCSS')
    plt.show()



def Kmeans_KMedoids_cluster(numerical_dataframe, n_clusters):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn_extra.cluster import KMedoids

    # Extract relevant features for clustering
    X = numerical_dataframe[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Perform K-Means clustering with chosen number of clusters (5 in this case)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_train)
    kmedoids.fit(X_train)
    y_kmeans = kmeans.predict(X_test)
    y_kmedoids = kmedoids.predict(X_test)

    # Visualize the clusters in 3D plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    for i in range(n_clusters):
        ax2.scatter(X_test[y_kmedoids == i, 2], X_test[y_kmedoids == i, 1], X_test[y_kmedoids == i, 3], s=100, c=colors[i],
                   label='Cluster {}'.format(i + 1))
    for i in range(n_clusters):
        ax.scatter(X_test[y_kmeans == i, 2], X_test[y_kmeans== i, 1], X_test[y_kmeans == i, 3], s=100,
                    c=colors[i],
                    label='Cluster {}'.format(i + 1))

    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Age')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('K-Means Clustering')

    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Age')
    ax2.set_zlabel('Spending Score (1-100)')
    ax2.set_title('K-Medoids Clustering')
    ax.legend()
    ax2.legend()
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

    elbow_best_number_of_clusters(df)

    for i in range(5):
        print(f"The cluster of each point when there are {i+1} clusters: ", Kmeans_KMedoids_cluster(df, i+1))

