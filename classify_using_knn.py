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
import sklearn.metrics as metrics

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
print(patients_data.groupby('target').size() / patients_data.shape[0])

############################## Training and Testing the data ###########################################################


patients_data_train, patients_data_test = train_test_split(patients_data,random_state= 42, stratify=patients_data['target'])
# the random state parameter in split, if int, random_state is the seed used by the random number generator and as it
# is a fixed number the outcome of split will be deterministic means split for Run 1 = split for Run2, we can use any
# number  for random_state parameter like 0,42, 21, etc.
print(patients_data.columns)
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

###### Randomized PCA
#rnd_pca= PCA(n_components=5, svd_solver="randomized")
#X_reduced_train = pandas.DataFrame(rnd_pca.fit_transform(X_train))
#X_reduced_test = pandas.DataFrame(rnd_pca.fit_transform(X_test))

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
knn_prob = knn.predict_proba(X_reduced_test)
disease_prob_knn = knn_prob[:,1]
knn_pred = knn.predict(X_reduced_test)

################################### Implementing Decision tree after Dimensionality Reduction  #########################
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

heart_disease_DT = classTree.fit(X_reduced_train, y_train)
print("Train set score after dimensionality reduction using Decision Tree as classifier: {:.2f}".format(
    classTree.score(X_reduced_train, y_train)))
print("Test set score after dimensionality reduction using Decision Tree as classifier: {:.2f}".format
      (classTree.score(X_reduced_test, y_test)))
DT_prob = classTree.predict_proba(X_reduced_test)
disease_DT_prob = DT_prob[:,1]
DT_pred =classTree.predict(X_reduced_test)


heart_disease_DT_without_PCA = classTree.fit(X_train, y_train)
dot_data = tree.export_graphviz(heart_disease_DT,out_file=None, impurity=True, filled=True,
                                feature_names=["Age","Sex","Chest Pain Type","Resting Blood Pressure","Serum Cholestrol"
                                    ,"Fasting Blood Sugar","Resting Electrocardiographic Results",
                                               "Maximum Heart Rate Achieved","Exercise Induced Angina ",
                                               "ST depression induced by exercise relative to rest",
                                               "Slope of the peak exercise ST segment",
                                            "Number of major vessels",
                                               "THAL"] , class_names=['Not Present','Present'])
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
lr_prob = logit.predict_proba(X_reduced_test)
disease_lr_prob = lr_prob[:,1]
lr_pred = logit.predict(X_reduced_test)
'''
Results are as follows for accuracy -  
Train set score after dimensionality reduction using KNN as classifier: 0.85
Test set score after dimensionality reduction using KNN as classifier: 0.84
Train set score after dimensionality reduction using Decision Tree as classifier: 0.91
Test set score after dimensionality reduction using Decision Tree as classifier: 0.75
Train set score after dimensionality reduction using Multinomial Logistic Regression as classifier: 0.85
Test set score after dimensionality reduction using Multinomial Logistic Regression as classifier: 0.83
'''

def evaluation_metrics(predProbY,nY, Y, pred_by_model):
    # Determine the predicted class of Y
    true_y_val = Y.values
    predY = np.zeros(nY)
    for i in range(nY):
        if predProbY[i]>= 0.5:
            predY[i] = 1
        else:
            predY[i] = 0

    # Calculate the Root Average Squared Error
    RASE = 0.0
    for i in range(nY):
        if true_y_val[i] ==1:
            RASE += (1 - predProbY[i])**2
        else:
            RASE += (0 - predProbY[i])**2

    RASE = np.sqrt(RASE/nY)
    RASE = np.sqrt(metrics.mean_squared_error(true_y_val, predProbY))
    AUC = metrics.roc_auc_score(true_y_val, pred_by_model)
    accuracy = metrics.accuracy_score(Y, predY)
    print('                  Accuracy: {:.13f}'.format(accuracy))
    print('    Misclassification Rate: {:.13f}'.format(1 - accuracy))
    print('          Area Under Curve: {:.13f}'.format(AUC))
    print('Root Average Squared Error: {:.13f}'.format(RASE))




print("Metrics for K-Neares-Neighbors")
evaluation_metrics(disease_prob_knn, X_test.shape[0],y_test, knn_pred)
print("Metrics for Decision Tree")
evaluation_metrics(disease_DT_prob,X_test.shape[0],y_test, DT_pred)
print("Metrics for Logistic Regression")
evaluation_metrics(disease_lr_prob, X_test.shape[0],y_test, lr_pred)


'''
Metrics for K-Neares-Neighbors
                  Accuracy: 0.8421052631579
    Misclassification Rate: 0.1578947368421
          Area Under Curve: 0.8369337979094
Root Average Squared Error: 0.3655036399936
Metrics for Decision Tree
                  Accuracy: 0.7500000000000
    Misclassification Rate: 0.2500000000000
          Area Under Curve: 0.7473867595819
Root Average Squared Error: 0.4330595232888
Metrics for Logistic Regression
                  Accuracy: 0.8289473684211
    Misclassification Rate: 0.1710526315789
          Area Under Curve: 0.8205574912892
Root Average Squared Error: 0.3440307449803
'''
###################################  Receiver Operating Charecteristics (ROC) curve of three models ####################

def roc_curve_generator(Y, pred_prob_knn, pred_prob_dt, pred_prob_lr):
    cutoffs = [2., 1., 0.72, 0.49230769, 0.3,
               0.23076923, 0.19298246, 0.17105263, 0.13043478,
               0.09090909, -1]   # Generate the coordinates
    knn_sensitivity = []
    knn_one_minus_specificity = []
    logistc_sensitivity = []
    logistic_one_minus_specificity = []
    dt_sensitivity = []
    dt_one_minus_specificity = []
    for cutoff in cutoffs:
        knn_y_pred = np.zeros(Y.count())
        dt_y_pred = np.zeros(Y.count())
        logistic_y_pred = np.zeros(Y.count())

        knn_y_pred[pred_prob_knn >= cutoff] = 1
        dt_y_pred[pred_prob_dt >= cutoff] = 1
        logistic_y_pred[pred_prob_lr >= cutoff] = 1

        true_neg, false_pos, false_neg, true_pos = metrics.confusion_matrix(Y, knn_y_pred).ravel()
        knn_sensitivity.append(true_pos / (true_pos + false_neg))
        knn_one_minus_specificity.append(false_pos / (false_pos + true_neg))

        true_neg, false_pos, false_neg, true_pos = metrics.confusion_matrix(Y, dt_y_pred).ravel()
        dt_sensitivity.append(true_pos / (true_pos + false_neg))
        dt_one_minus_specificity.append(false_pos / (false_pos + true_neg))

        true_neg, false_pos, false_neg, true_pos = metrics.confusion_matrix(Y, logistic_y_pred).ravel()
        logistc_sensitivity.append(true_pos / (true_pos + false_neg))
        logistic_one_minus_specificity.append(false_pos / (false_pos + true_neg))

    # Draw the ROC curve
    axs = plt.gca()
    # plt.figure(figsize=(6,6))
    axs.plot(knn_one_minus_specificity, knn_sensitivity, marker='o',
             color='purple', linestyle='solid', linewidth=2, markersize=6, label="K-Nearest-Neaighbors")
    axs.plot(dt_one_minus_specificity, dt_sensitivity, marker='o',
                 color='blue', linestyle='solid', linewidth=2, markersize=6, label="Decision Tree")
    axs.plot([0, 1], [0, 1], color='red', linestyle=':')
    axs.plot(logistic_one_minus_specificity, logistc_sensitivity, marker='o',
                 color='green', linestyle='solid', linewidth=2, markersize=6, label="Logistic Regression")

    axs.legend()
    plt.grid(True)
    plt.xlabel("1 - Specificity (False Positive Rate)")
    plt.ylabel("Sensitivity (True Positive Rate)")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()


roc_curve_generator(y_test, disease_prob_knn, disease_DT_prob, disease_lr_prob)