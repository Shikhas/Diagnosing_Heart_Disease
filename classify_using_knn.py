import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

########################################## Meet the data ###############################################################
# 303 patient data
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

# scikit-learn shuffles data using pseudorandom number generator and splits it into 75% training data and 25% testing
# data. Shuffling the data is important so that test data contains data from all classes. scikit-learn uses capital X
# for data and lowecase y for labels.

X_train, X_test, y_train, y_test = train_test_split(X_data,Y_data,random_state= 42)
# the random state parameter in split, if int, random_state is the seed used by the random number generator and as it
# is a fixed number the outcome of split will be deterministic means split for Run 1 = split for Run2, we can use any
# number  for random_state parameter like 0,42, 21, etc.print(X_data.columns)
print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", X_test.shape)

################################### Look at our data ###################################################################
# we can visualize data by plotting a pair plot of features also called scatter plot, the data points are colored
# according to the presence/absence of disease

#import ipdb; ipdb.set_trace()

# first normalize the data to maximize variance and change the scale of values into range of [0,1] while while
# standardizing, we change variables means to 0 and standard deviation to 1

# Normalization" onto [0,1] is called "feature scaling" or "unity-based normalization" on the Wikipedia page.
# "Normalization" based on the observed mean and standard deviation (called "Student's t-statistic" on that page;
# "standardization" in more frequent but not universal usage) is typically what you want for PCA.

# Feature scaling is a method used to standardize the range of independent variables or features of data.
# Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not
# work properly without normalization. For example, the majority of classifiers calculate the distance between two
# points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by
# this particular feature. Therefore, the range of all features should be normalized so that each feature contributes
# approximately proportionately to the final distance.
# Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling
# than without it.

# Few useful links for it
# https://en.wikipedia.org/wiki/Feature_scaling
# https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
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
# Dimensionality Reduction plays a really important role in machine learning, especially when you are working with
# thousands of features. Principal Components Analysis are one of the top dimensionality reduction algorithm, it is
# not hard to understand and use it in real projects. This technique, in addition to making the work of feature
# manipulation easier, it still helps to improve the results of the classifier

pca = PCA().fit(X_train)
print(pca.explained_variance_) #Explained variance is amount of variance explained by each of the selected component.
# In other words, it is the eigenvalue

print(pca.explained_variance_ratio_) #Explained Variance Ratio is Explained Variance divided by sum of eigenvalues
print()
print(X_train.columns.values.tolist())
print(pca.components_) #This shows the principal components where each column contains the coefficients for the linear
# transformation and the order of columns corresponds to that of Explained Variance