'''
Program Purpose and Workflow:
+ FIXME
'''
##########################################################
## Print Message Function
##########################################################
def print_message(string):
    print('#'*(len(string) + 2))
    print('#'+string+'#')
    print('#'*(len(string) + 2))

##Imports
print_message("Importing Necessary Libraries")
print("Importing Non-Sklearn Libraries")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import zscore
import itertools


##Sklearn Imports
print("Importing Sklearn Libraries")
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix
print_message("Imports Complete!")

##########################################################
##Plot Confusion Matrix Function
##########################################################

##From sklearn
##http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, fname = ""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    ##Function has option to show a normalized confusion matrix,
    ##which will show the relative percentages, rather than raw counts,
    ##of model predictions in each outcome class
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    ##Make figure for creating confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    ##Labeling the confusion matrix
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(fname)
##########################################################



##########################################################
##Loading, Concatenating, and Cleaning the Data for Analysis
##########################################################

##Load in data
print("\n\n")
print_message("Loading in the Dataset")
##Change the pathname and name of the datafile being used for analysis
data_df = pd.read_csv("../data.csv") ##Local Development Copy

##Extract Class Predictions and make them discrete integers
print("Transforming Classes Into Integers for the Model")
##Get health labels from the dataset, array of patients' health labels
labels = data_df['Health'].values
##Get unique
unique_labs = np.unique(labels)
y_true = []
##Assign value to patient's health label based on what its corresponding index
##is in the unique array
for i in labels:
    y_true.append([j for j in range(len(unique_labs)) if unique_labs[j] == i][0])

##Extract the indices for the tags for the bacterial/viral data
print("Getting the bacterial/viral counts from the data and clinical symptoms")
##Change these to correspond to the first and last column of the dataset that
##will be extracted for doing the microbio analysis
first_tag = 'Bacteroidetes'
last_tag = 'Virus'
##Change the corresponding column name into an index
first_loc = [x for x in range(len(data_df.columns)) if data_df.columns[x] == first_tag][0]
last_loc = [x+1 for x in range(len(data_df.columns)) if data_df.columns[x] == last_tag][0]

##Get the names of those features from the data
print("Getting feature names from data")
micro_bio_colums = data_df.columns[first_loc:last_loc]

##Getting the clinical columns
##Change these to correspond to the first and last column of the dataset that
##will be extracted for doing the clinical symptoms analysis
first_tag = 'No Symptoms'
last_tag = 'Fever'
##Change the column name into an index and extract feature names
first_loc = [x for x in range(len(data_df.columns)) if data_df.columns[x] == first_tag][0]
last_loc = [x+1 for x in range(len(data_df.columns)) if data_df.columns[x] == last_tag][0]
clinical_columns = data_df.columns[first_loc:last_loc]

print_message("Data Selected ")
##Extract the matrix of microbio expression data and normalize
##Log2 transform and the z-score normalization
micro_bio_data = data_df[micro_bio_colums].values.astype(float)
micro_bio_data = np.log2(micro_bio_data)
##May get runtime warning for divide by zero in log2, this is something that
##can be ignored. First, log2 transform the data
##All of the "NaN" and "inf" values are turned into a -1, which will be the
##default null value
micro_bio_data[np.isnan(micro_bio_data)] = -1
micro_bio_data[np.isinf(micro_bio_data)] = -1
##Once all of the values are log2 transformed and corrected, then do a zscore
##transformation, subtract the mean and divide by the standard deviation,
##Along each of the columns, or each of the features across all patients,
##so values in features will be relative to that feature in every other patient
micro_bio_data = zscore(micro_bio_data, axis = 1)
micro_bio_data[np.isnan(micro_bio_data)] = -1

##Handling the string values for the clinical symptoms
try:
    ##If the values are already in a format that can be converted to a float
    clinical_data = data_df[clinical_columns].values.astype(float)
except:
    ##Otherwise, take in the values as a string
    clinical_data = data_df[clinical_columns].values.astype(str)

    ##Function to select a small subset of the values in each of the symptoms,
    ##which is just indexing the strings
    def subset(item):
        return item.split(" ")[0].lower()

    ##Vectorize the function to apply to array of values
    subset_array = np.vectorize(subset)

    ##Now, subset each string in the array
    clinical_data = subset_array(clinical_data)

    ##Turn "nan" and "missing" datapoints into -1
    clinical_data[clinical_data == 'nan'] = "-1"
    clinical_data[clinical_data == 'missing'] = "-1"

    ##Yes is the positive symptom indicator and is marked by a 1
    clinical_data[clinical_data == 'yes'] = "1"

    ##No is the negative symptom marker and is noted with a 0
    clinical_data[clinical_data == 'no'] = "0"

    ##Convert clinical data to floats now
    clinical_data = clinical_data.astype(float)

##Total Features collected
features = np.array(list(clinical_columns)+list(micro_bio_colums))

##Final Model Data
X = np.concatenate((clinical_data,micro_bio_data), axis = 1)
y = y_true

print_message("Data Loaded and Ready!!")
##########################################################
##########################################################

print('\n')

##########################################################
##Random Forest Classifier
##########################################################

print_message("Working on RFC")

##Make dictionary to store model importances
model_importances = {}
##Take combined list of features for clinical features and microbio features
for feature in list(clinical_columns)+list(micro_bio_colums):
    ##Set each feature's importance to 0
    model_importances[feature] = 0

##Store AUC for cross validations
auc_validations = {}
##For each unique class in the dataset
for k in np.unique(y_true):
    ##Create an empty list to store the AUC for each iteration in cross-validation
    auc_validations[k] = []

##K-fold Cross Validation, 20 iterations
for T in range(20):
    ##Train-Test Split the aggregate data with Test Size of 25%
    X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.25)

    ##Train the Model, n_estimators set to 30, 5 jobs run
    rfc = RFC(n_estimators=30, n_jobs = 5)
    ##Fir the model
    rfc.fit(X_train, y_train)
    ##Get continuous probabiltiy prediction and boolean prediction
    y_pred = rfc.predict_proba(X_test)
    y_pred_b = rfc.predict(X_test)

    ##Create encoding for AUC
    y_test_e = label_binarize(y_test, classes = np.unique(y_test))

    ##Compute ROC Curve and AUC for each class
    ##False positive rate dictionary
    fpr = dict()
    ##True positive rate dictionary
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test_e.shape[1]):
        ##Calculate ROC and AUC
        fpr[i], tpr[i], _ = roc_curve(y_test_e[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    ##Store the AUC for each class in the dictionary defined before
    for k in roc_auc.keys():
        auc_validations[k].append(roc_auc[k])


    ##Extract Feature Importances from RF model
    importances = rfc.feature_importances_
    ##Sort the features from most important to least important using argsort
    sorted_inds = np.argsort(importances)[::-1]
    ##Sort the features
    sorted_features = features[sorted_inds]
    sorted_importances = importances[sorted_inds]

    ##For each of the features
    for i,val in enumerate(sorted_features):
        ##Update the total feature importance in the dictionary
        ##for feature importances
        model_importances[val] += sorted_importances[i]

    ##On the final run, print out a single confusion Matrix
    ##to verify the model's performance, save the figure to the Results
    ##directory, print a classification report. Meant as a final check of
    ##behavior, not as a definitive result
    if T==19:
        rfc = plot_confusion_matrix(confusion_matrix(y_test, y_pred_b), classes = ["Sick", "Healthy", "Follow"], fname = "../results/random_forest.png")
        print(confusion_matrix(y_test, y_pred_b))
        print(classification_report(y_test, y_pred_b))


##Print Results for Random Forest
print_message("Random Forest Classifier")
##Print out the AUC for each fo the class after K fold generation
print('\n'+"After {} Trials:".format(T+1))
for k in auc_validations.keys():
    print("AUC {}: {}".format(unique_labs[k], round(np.mean(auc_validations[k]),4)))

##Final Feature Importances Plot
##This is the main motivation of using the Random Forest model. We wanted to see
##if there was some kind of match between the feature importances of the model
##and what we see in the literature, plot the final aggregate feature importances
##and send the plot out to the results directory
importances = np.array(list(model_importances.values()))
sorted_inds = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_inds]/len(importances)
features = np.array(list(model_importances.keys()))
sorted_features = features[sorted_inds]
plt.figure(figsize=(13, 7))
ax = plt.gca()
ax.bar(range(len(sorted_inds)), sorted_importances)
ax.set_title("Feature Importances", fontsize = 23)
ax.set_ylabel("Feature Importances", fontsize = 16)
ax.set_xlabel("Feature", fontsize = 16)
ax.set_xticks(range(len(sorted_inds)))
ax.set_xticklabels(sorted_features, rotation = 90)
plt.tight_layout()
plt.savefig("../results/full_w_clinical_importances.png", dpi = 200, bbox_inches = 'tight')

##Save final array of feature importances. The order is in the same order as
##the column list when creating it, clinical features and then microbio features
to_save = np.array(importances)
np.savetxt("../results/importances_in_order_of_features.txt", to_save)

##########################################################

print('\n')

##Conduct the same thing with Linear SVM and Logistic Regression

##########################################################
##Linear SVM
##########################################################
print_message("Working on Linear Support Vector Machine")

##Store AUC for CV
auc_validations = {}
for k in np.unique(y_true):
    auc_validations[k] = []

##Cross Validation by Repeat Trials
for T in range(20):
    # print("Trial {}".format(T+1))
    ##Train-Test Split
    X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.25)

    ##Train the Model
    ##Using the OneVsRestClassifier with a Linear Support Vector Machine
    ##the kernel is set to linear. Use the default parameters the exception
    ##of the kernel and setting probabilities to true, this return probabilities
    ##for each class
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    classifier.fit(X_train, y_train)

    ##Prediction in a continuous form and boolean form
    y_pred = classifier.predict_proba(X_test)
    y_pred_b = classifier.predict(X_test)

    ##Create encoding for AUC
    y_test_e = label_binarize(y_test, classes = np.unique(y_true))

    # Compute ROC Curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test_e.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_e[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for k in roc_auc.keys():
        auc_validations[k].append(roc_auc[k])

    ##On the final iteration, plot a confusion matrix and return a classification
    ##report to verify the model's behavior, save the matrix to the Results
    ##directory
    if T==19:
        lin_reg = plot_confusion_matrix(confusion_matrix(y_test, y_pred_b), classes = ["Sick", "Healthy", "Follow"], fname = "../results/lin_svm.png")
        print(confusion_matrix(y_test, y_pred_b))
        print(classification_report(y_test, y_pred_b))

##Print Results for the AUC
print_message("Linear SVM")
print('\n'+"After {} Trials:".format(T+1))
for k in auc_validations.keys():
    print("AUC {}: {}".format(unique_labs[k], round(np.mean(auc_validations[k]),4)))

##########################################################

##Perform the same pipeline with the Logistic Regression

print('\n')

##########################################################
##Logistic Regression
##########################################################
print_message("Working on Logistic Regression")

##Store AUC for CV
auc_validations = {}
for k in np.unique(y_true):
    auc_validations[k] = []

##Cross Validation by Repeat Trials
for T in range(20):
    ##Train-Test Split
    X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.25)

    ##Train the Model
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict_proba(X_test)
    y_pred_b = classifier.predict(X_test)

    ##Create encoding for AUC
    y_test_e = label_binarize(y_test, classes = np.unique(y_true))

    # Compute ROC Curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test_e.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_e[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for k in roc_auc.keys():
        if roc_auc[k] != np.nan:
            auc_validations[k].append(roc_auc[k])

    if T==19:
        log_reg = plot_confusion_matrix(confusion_matrix(y_test, y_pred_b), classes = ["Sick", "Healthy", "Follow"], fname = "../results/log_reg.png")
        print(confusion_matrix(y_test, y_pred_b))
        print(classification_report(y_test, y_pred_b))

##Print Results
print_message("Logistic Regression")
print('\n'+"After {} Trials:".format(T+1))
for k in auc_validations.keys():
    print("AUC {}: {}".format(unique_labs[k], round(np.mean(auc_validations[k]),4)))

##########################################################


##########################################################
##Comaprison to Methods KMeans and the different models
##########################################################
##K values to try in kmeans clustering
##The motivation for this is that we are hoping to be able to find some kind of
##agreement between the number of clusters used for clustering and the number
##of classes in the dataset, as well as seeing where in the dataset kmeans
##agrees with the different models
ks = [2,3,4,5,6,7,8,9,10]

##Run pca take first 5 principle components of the dataset to reduce the number
##of features in the dataset
X = PCA(n_components = 5).fit_transform(X)

##For each K in the K Means trials
for K in ks:

    ##Store Adjusted Rand Indices for the different comparisons, it will compare
    ##the two outputs for the different models and assign a value for comparison
    rand_indexes = {"Log,RFC":0, "Log,Lin":0, "RFC,Lin":0, "RFC,KMeans":0, "Log,KMeans":0, "KMeans,Lin":0}

    print_message("Comparison Results for K={}".format(K))

    ##Cross Validation by Repeat Trials, K-fold cross validation, 20 times
    cnt = 1
    for T in range(20):

        ##Train-Test Split
        X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.25)

        ##Run each of the models, since nothing here is changing,
        ##then each run of the K Means with varying k values is another version
        ##of cross-validation for RFC, Linear SVM, and Logistic Regression. Each
        ##run will be a consistency check for behavior

        ##Logistic Regression
        lr_classifier = LogisticRegression()
        y_pred_log = lr_classifier.fit(X_train, y_train).predict(X_test)

        ##Random Forest
        rfc = RFC(n_estimators=30, n_jobs = 5)
        rfc.fit(X_train, y_train)
        y_pred_rf = rfc.predict(X_test)

        ##KMeans
        ##Assigns new data to the nearest centroid
        kmeans = KMeans(n_clusters = K)
        kmeans.fit(X_train, y_train)
        y_pred_km = kmeans.predict(X_test)

        ##Linear SVM
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
        y_pred_lin = classifier.fit(X_train, y_train).predict(X_test)

        ##Add to each of the dictionary keys. Adding the Adjusted rand index for
        ##each of the trials to get an aggregate adjusted rand index
        rand_indexes["Log,RFC"] += adjusted_rand_score(y_pred_log, y_pred_rf)
        rand_indexes["Log,KMeans"] += adjusted_rand_score(y_pred_log, y_pred_km)
        rand_indexes["Log,Lin"] += adjusted_rand_score(y_pred_log, y_pred_lin)
        rand_indexes["RFC,KMeans"] += adjusted_rand_score(y_pred_rf, y_pred_km)
        rand_indexes["RFC,Lin"] += adjusted_rand_score(y_pred_rf, y_pred_lin)
        rand_indexes["KMeans,Lin"] += adjusted_rand_score(y_pred_km, y_pred_lin)
        cnt += 1

    ##Print Result
    ##After the 20 trials, then print out the average adjusted rand indices for
    ##each of the comparison methods
    print("Result for K = {}".format(K))
    for key in rand_indexes.keys():
        print("{}: {}".format(key, round(rand_indexes[key]/cnt, 4)))
    print("\n")
##########################################################
