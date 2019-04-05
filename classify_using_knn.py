import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

########################################## Meet the data ###############################################################
patients_data = pandas.read_csv("heart.csv")

print("First 5 rows are:\n",patients_data.head())
print("Last 5 rows are:\n",patients_data.tail())
print("The summary of the patients data is : \n", patients_data.describe())
print("The columns : \n", patients_data.columns) #listing all the columns
# print("The age column data \n", patients_data["age"])
print("The shape of the dataset is {}".format(patients_data.shape))
print("The type of target is {}".format(type(patients_data["age"])))
# print("The target data \n", patients_data["target"]) # viewing the target - 0 or 1 if disease is absent then target
# value is 0 otherwise 1

############################## Training and Testing the data ###########################################################
X_data = patients_data.drop('target',axis=1) # 1 is for column name in axis
Y_data = patients_data['target']

X_train, X_test, y_train, y_test = train_test_split(X_data,Y_data,random_state= 42)
# the random state parameter in split, if int, random_state is the seed used by the random number generator and as it
# is a fixed number the outcome of split will be deterministic means split for Run 1 = split for Run2, we can use any
# number  for random_state parameter like 0,42, 21, etc.
# print(X_data.columns)
print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", X_test.shape)

################################### Look at our data ###################################################################
# we can visualize data by plotting a pair plot of features also called scatter plot, the data points are colored
# according to the presence/absence of disease

#import ipdb; ipdb.set_trace()

X_train=(X_train-np.mean(X_train))/(np.std(X_train)).values  #Standardization
#X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values  #Rescaling (min-max normalization)
X_test=(X_test-np.mean(X_test))/(np.std(X_test)).values

df = pandas.DataFrame(X_train, columns=X_data.columns)
scatter_matrix(df, c= y_train,alpha= 0.8,figsize=(15,15),diagonal='hist', s = 200 ,marker='o')
plt.show()

# instead of histograms for visualizing density we can have probability distribution function by using kde
scatter_matrix(df, c= y_train,alpha= 0.8,figsize=(15,15),diagonal='kde', s = 200 ,marker='o')
plt.show()

################################### Dimensionality Reduction  ##########################################################
nVar = X_train.shape[1]
pca = PCA().fit(X_train)
print(pca.explained_variance_) #Explained variance is amount of variance explained by each of the selected component.
# In other words, it is the eigenvalue

print(pca.explained_variance_ratio_) #Explained Variance Ratio is Explained Variance divided by sum of eigenvalues
print()
print(X_train.columns.values.tolist())
#print(pca.components_) #This shows the principal components where each column contains the coefficients for the linear
# transformation and the order of columns corresponds to that of Explained Variance

cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)


plt.plot(pca.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

cumsum_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(np.arange(0,nVar))
plt.grid(True)
plt.show()


#Links - https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c

################################### Inplementing knn without Dimensionality Reduction  #################################
knn = KNeighborsClassifier(n_neighbors=2)  #n_neighbors is number of neighbors to use for kneighbors queries
knn.fit(X_train, y_train)
print("Train set score: {:.2f}".format(knn.score(X_train, y_train)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# Without PCA we achieve 80% accuracy for test data and 88% for traning data