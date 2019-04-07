import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import graphviz
import statsmodels.api as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


patients_data_train, patients_data_test = train_test_split(patients_data,random_state= 42, stratify=patients_data['target'])
# the random state parameter in split, if int, random_state is the seed used by the random number generator and as it
# is a fixed number the outcome of split will be deterministic means split for Run 1 = split for Run2, we can use any
# number  for random_state parameter like 0,42, 21, etc.
# print(X_data.columns)
X_train = patients_data_train.drop('target',axis=1) # 1 is for column name in axis
y_train = patients_data_train['target']

X_test = patients_data_test.drop('target',axis=1) # 1 is for column name in axis
y_test = patients_data_test['target']

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

################################### Look at our data ###################################################################
# we can visualize data by plotting a pair plot of features also called scatter plot, the data points are colored
# according to the presence/absence of disease

#import ipdb; ipdb.set_trace()

X_train=(X_train-np.mean(X_train))/(np.std(X_train)).values  #Standardization
#X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values  #Rescaling (min-max normalization)
X_test=(X_test-np.mean(X_test))/(np.std(X_test)).values
'''
df = pandas.DataFrame(X_train, columns=X_data.columns)
scatter_matrix(df, c= y_train,alpha= 0.8,figsize=(15,15),diagonal='hist', s = 200 ,marker='o')
plt.show()

# instead of histograms for visualizing density we can have probability distribution function by using kde
scatter_matrix(df, c= y_train,alpha= 0.8,figsize=(15,15),diagonal='kde', s = 200 ,marker='o')
plt.show()
'''
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


plt.plot(np.arange(1,nVar+1), pca.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1,nVar+1))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

plt.plot(np.arange(1,nVar+1),pca.explained_variance_,marker = 'o')
plt.xlabel("Index")
plt.ylabel("Explained Variance")
plt.xticks(np.arange(1,nVar+1))
plt.grid(True)
plt.show()

plt.plot(np.arange(1,nVar+1),cumulative_var_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(np.arange(1,nVar+1))
plt.grid(True)
plt.show()

# pca = PCA(n_components=0.90) we can set n_components to be a float between 0.0 to 1.0, indicating the ratio of
# variance we wish to preserve
pca = PCA(n_components=5)  # as the elbow in the graph is at number of dimensions to be 5
X_reduced_train = pandas.DataFrame(pca.fit_transform(X_train))
X_reduced_test = pandas.DataFrame(pca.fit_transform(X_test))

################################### Inplementing knn without Dimensionality Reduction  #################################
knn = KNeighborsClassifier(n_neighbors=13)  #n_neighbors is number of neighbors to use for kneighbors queries
knn.fit(X_train, y_train)
print("Train set score without dimensionality reduction using KNN: {:.2f}".format(knn.score(X_train, y_train)))
print("Test set score without dimensionality reduction using KNN: {:.2f}".format(knn.score(X_test, y_test)))

n_neighbors = range(1, 20)
train_data_accuracy = []
test_data_accuracy = []
for n_neigh in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=n_neigh,n_jobs=5)
    knn.fit(X_reduced_train, y_train)
    train_data_accuracy.append(knn.score(X_reduced_train, y_train))
    test_data_accuracy.append(knn.score(X_reduced_test, y_test))
plt.plot(n_neighbors, train_data_accuracy, label="Train Data Set")
plt.plot(n_neighbors, test_data_accuracy, label="Test Data Set")
plt.ylabel("Accuracy")
plt.xlabel("Neighbors")
plt.legend()
plt.show()

n_neighbors = range(1, 20)
k_scores=[]
for n_neigh in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=n_neigh,n_jobs=5)
    scores=cross_val_score(estimator=knn,X=X_train,y=y_train,cv=12)
    k_scores.append(scores.mean())
print(k_scores)

plt.plot(n_neighbors, k_scores)
plt.xlabel("Value of k for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.show()

################################### Implementing knn after Dimensionality Reduction  #################################
knn = KNeighborsClassifier(n_neighbors=14)  #n_neighbors is number of neighbors to use for kneighbors queries
knn.fit(X_reduced_train, y_train)
print("Train set score after dimensionality reduction using KNN as classifier: {:.2f}".format(knn.score
                                                                                            (X_reduced_train, y_train)))
print("Test set score after dimensionality reduction using KNN as classifier: {:.2f}".format(knn.score
                                                                                            (X_reduced_test, y_test)))

################################### Implementing Decision tree after Dimensionality Reduction  #########################
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

heart_disease_DT = classTree.fit(X_reduced_train, y_train)
print("Train set score after dimensionality reduction using Decision Tree as classifier: {:.2f}".format(
    classTree.score(X_reduced_train, y_train)))
print("Test set score after dimensionality reduction using Decision Tree as classifier: {:.2f}".format
      (classTree.score(X_reduced_test, y_test)))

heart_disease_DT_without_PCA = classTree.fit(X_train, y_train)
dot_data = tree.export_graphviz(heart_disease_DT,out_file=None, impurity=True, filled=True,
                                feature_names= X_train.columns, class_names=['0','1'])
graph = graphviz.Source(dot_data)
graph.render('heart_disease_decision_tree',".")

################################### Implementing Multinomial Logit after Dimensionality Reduction  #####################
'''
# Multinomial Logit takes input as array
X_reduced_train = stats.add_constant(X_reduced_train, prepend=True)
X_reduced_test = stats.add_constant(X_reduced_test,prepend=True)

logit = stats.MNLogit(y_train.values ,X_reduced_train.values)
thisFit = logit.fit(method='newton', full_output=True, maxiter=100)
thisParameter = thisFit.params
print("Train set score after dimensionality reduction using Multinomial Logistic Regression as classifier: {:.2f}".format(
    logit.score(X_reduced_train.as_matrix)))
print("Test set score after dimensionality reduction using Multinomial Logistic Regression as classifier: {:.2f}".format
      (logit.score(X_reduced_test.as_matrix)))
'''
logit = LogisticRegression()
logit.fit(X_reduced_train, y_train)
print("Train set score after dimensionality reduction using Multinomial Logistic Regression as classifier: {:.2f}".format(
    logit.score(X_reduced_train,y_train)))
print("Test set score after dimensionality reduction using Multinomial Logistic Regression as classifier: {:.2f}".format
      (logit.score(X_reduced_test,y_test)))

'''
Train set score after dimensionality reduction using KNN as classifier: 0.85
Test set score after dimensionality reduction using KNN as classifier: 0.84
Train set score after dimensionality reduction using Decision Tree as classifier: 0.91
Test set score after dimensionality reduction using Decision Tree as classifier: 0.75
Train set score after dimensionality reduction using Multinomial Logistic Regression as classifier: 0.85
Test set score after dimensionality reduction using Multinomial Logistic Regression as classifier: 0.83
'''