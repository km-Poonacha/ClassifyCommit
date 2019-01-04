# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:30:42 2019

@author: kmpoo
"""

import pandas as pd
import numpy as np
import ast
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle #To shuffle the dataframe
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


TRAIN_CSV = 'C:/Users/kmpoo/Dropbox/HEC/Project 5 - Roles and Coordination/Data/ML/Commit Creativity - Train2.csv'
LABELFULL_CSV = 'C:/Users/kmpoo/Dropbox/HEC/Project 5 - Roles and Coordination/Data/ML/Trainout.csv'
TRAINSET_CSV = 'C:/Users/kmpoo/Dropbox/HEC/Project 5 - Roles and Coordination/Data/ML/Trainset.csv'
TESTSET_CSV = 'C:/Users/kmpoo/Dropbox/HEC/Project 5 - Roles and Coordination/Data/ML/Testset.csv'

def plot_learning_curve_std(estimator, X, y):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    train_sizes, train_scores, test_scores = learning_curve( RandomForestClassifier(), 
                                                            X, 
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv= 10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=1, 
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.001, 1.0, 20))
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    
    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    

def getnoelements(x):
    """ Parse str represetation of python object into a list and return lenght"""
    no_parents = len(ast.literal_eval(x))
    return no_parents
    
def dataclean(dataframe):
    """Various cleaning done for trainig data"""
    dataframe = dataframe.drop(axis=1,columns=['Commit URL', 'Sha','URL','Committer Name','Committer Email','Commit Date ','Verification','Author Date'])
    dataframe = shuffle(dataframe)
    # Encode type of commit
    dataframe = dataframe.assign(CommitType = lambda x: x['Type of Commit (Primary)'].str.split().str.get(0).str.strip(','))
    dataframe.CommitType = dataframe.CommitType.replace({'Feature': 1,
                                                                       'Bug/Issue': 2,
                                                                       'Documentation': 3,
                                                                       'Peer': 4,
                                                                       'Process': 5,
                                                                       'Testing': 6})
    dataframe = dataframe.drop(axis=1,columns=['Type of Commit (Primary)','Optional Type of Commit (Secondary)'])
    # Find number of parents
    dataframe['Parents'] = dataframe['Parents'].map(getnoelements)
    # Convert the number of lines of code into nChanges, nAdditions, nDeletions
    # Get total number of changes
    dataframe = dataframe.assign(nChanges = lambda x : x['Lines of Code Changed'].str.split(':').str.get(1).str.split(',').str.get(0) )
    # Get total number of additions
    dataframe = dataframe.assign(nAdditions = lambda x : x['Lines of Code Changed'].str.split(':').str.get(2).str.split(',').str.get(0) )
    # Get total number of deletions
    dataframe = dataframe.assign(nDeletions = lambda x : x['Lines of Code Changed'].str.split(':').str.get(3).str.split('}').str.get(0) )
    # Create three class labeld for novelty and usefulness
    conditions = [
        (dataframe['Novelty '] > 3),
        (dataframe['Novelty '] < 3),]
    choices = ['High', 'Low']
    dataframe['Novelty3'] = np.select(conditions, choices, default='Medium')
    
    conditions = [
        (dataframe['Usefulness '] > 3),
        (dataframe['Usefulness '] < 3),]
    choices = ['High', 'Low']
    dataframe['Usefulness3'] = np.select(conditions, choices, default='Medium')
    #Create count of words feature
    dataframe = dataframe.assign(nWords = lambda x : x['Description'].str.split().str.len() )
    dataframe_sd = dataframe.drop(dataframe[dataframe.CommitType.astype(int) == 3].index)

    return dataframe, dataframe_sd

def vectordsc(corpus, train_text, test_text):
    # convert dtype object to unicode
#    vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(corpus.values.astype('U')) #BOW unigram representaion 
    # X is a sparse matrix
    word_vectorizer = TfidfVectorizer(
                                        sublinear_tf=True,
                                        strip_accents='unicode',
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        stop_words='english',
                                        ngram_range=(1, 1),
                                        max_features=10000)
#    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
#    analyze = bigram_vectorizer.build_analyzer()
#    Y = bigram_vectorizer.fit_transform(corpus.values.astype('U')).toarray()
    word_vectorizer.fit(corpus)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    return train_word_features, test_word_features

def MLPmodel(train_x, train_y, test_x, test_y):
    nn = MLPClassifier(
                        hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
                        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    n = nn.fit(train_x, train_y)
    p_train = n.predict_proba(train_x)
    p_test = n.predict_proba(test_x)
    print("accuracy is = ", n.score(test_x,test_y))
    plot_learning_curve_std(nn, train_x, train_y)
    return p_train,p_test

def RFCmodel(train_x, train_y, test_x, test_y):
    rfc = RandomForestClassifier(n_estimators=10)
    r = rfc.fit(train_x, train_y)
    print("accuracy of rfc is = ", r.score(test_x,test_y))
    p_train = r.predict_proba(train_x)
    p_test = r.predict_proba(test_x)
    plot_learning_curve_std(rfc, train_x, train_y)
    return p_train,p_test

def SVMmodel(train_x, train_y, test_x, test_y):
    svc = svm.SVC(kernel='linear',probability=True)  
    s = svc.fit(train_x, train_y)
    print("accuracy of svm is = ", s.score(test_x,test_y))
    p_train = s.predict_proba(train_x)
    p_test = s.predict_proba(test_x)
    return p_train,p_test  


pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format

vector_dataframe = pd.read_csv(TRAIN_CSV, sep=",",error_bad_lines=False,header= 0, low_memory=False, encoding = "Latin1")

# Shuffle the dataframe
vector_dataframe, vector_dataframe_sd = dataclean(vector_dataframe)
vector_dataframe.to_csv(LABELFULL_CSV)
df_train, df_test = train_test_split(vector_dataframe, test_size=0.2)
df_train=df_train.reset_index()
df_test = df_test.reset_index()
t0 = time()
train_x, test_x = vectordsc(vector_dataframe['Description'], df_train['Description'], df_test['Description'] )
print("Time to extract features = ", time() - t0)
df_train.to_csv(TRAINSET_CSV)
df_test.to_csv(TESTSET_CSV)
train_y = df_train['Novelty ']


'''MLPClassifier'''
#Stage 1  
p_train,p_test = MLPmodel(train_x, df_train['Novelty '], test_x, df_test['Novelty '])
p_train2,p_test2 = MLPmodel(train_x, df_train['Novelty3'], test_x, df_test['Novelty3'])

#Stage 2
df_train_prob = pd.DataFrame(p_train, columns = ['p1','p2','p3','p4','p5'])
train_x_s2 = pd.concat([df_train_prob,df_train['Files Changed'],df_train['nAdditions'],df_train['nDeletions'],df_train['Parents'],df_train['nWords']], axis=1)
df_test_prob = pd.DataFrame(p_test, columns = ['p1','p2','p3','p4','p5'])
test_x_s2 = pd.concat([df_test_prob,df_test['Files Changed'],df_test['nAdditions'],df_test['nDeletions'],df_test['Parents'],df_test['nWords']], axis=1)

p_train_s2,p_test_s2 = MLPmodel(train_x_s2, df_train['Novelty3'], test_x_s2, df_test['Novelty3'])

# Single stage
train_x_1s = hstack((train_x,df_train['Files Changed'].astype(float).values[:, None], df_train['nAdditions'].astype(float).values[:, None], df_train['nDeletions'].astype(float).values[:, None], df_train['Parents'].astype(float).values[:, None] ,df_train['nWords'].astype(float).values[:, None]))
test_x_1s = hstack((test_x,df_test['Files Changed'].astype(float).values[:, None], df_test['nAdditions'].astype(float).values[:, None], df_test['nDeletions'].astype(float).values[:, None], df_test['Parents'].astype(float).values[:, None], df_test['nWords'].astype(float).values[:, None]))
p_train_1s,p_test_1s = MLPmodel(train_x_1s, df_train['Novelty '], test_x_1s, df_test['Novelty '])
p_train3_1s,p_test3_1s = MLPmodel(train_x_1s, df_train['Novelty3'], test_x_1s, df_test['Novelty3'])


'''Random Forest'''
print("************ Random Forest *************")
#Stage 1  
p_train,p_test = RFCmodel(train_x, df_train['Novelty '], test_x, df_test['Novelty '])
p_train2,p_test2 = RFCmodel(train_x, df_train['Novelty3'], test_x, df_test['Novelty3'])

#Stage 2
df_train_prob = pd.DataFrame(p_train, columns = ['p1','p2','p3','p4','p5'])
train_x_s2 = pd.concat([df_train_prob,df_train['Files Changed'],df_train['nAdditions'],df_train['nDeletions'],df_train['Parents'],df_train['nWords']], axis=1)
df_test_prob = pd.DataFrame(p_test, columns = ['p1','p2','p3','p4','p5'])
test_x_s2 = pd.concat([df_test_prob,df_test['Files Changed'],df_test['nAdditions'],df_test['nDeletions'],df_test['Parents'],df_test['nWords']], axis=1)

p_train_s2,p_test_s2 = RFCmodel(train_x_s2, df_train['Novelty3'], test_x_s2, df_test['Novelty3'])

# Single stage
train_x_1s = hstack((train_x,df_train['Files Changed'].astype(float).values[:, None], df_train['nAdditions'].astype(float).values[:, None], df_train['nDeletions'].astype(float).values[:, None], df_train['Parents'].astype(float).values[:, None] ,df_train['nWords'].astype(float).values[:, None]))
test_x_1s = hstack((test_x,df_test['Files Changed'].astype(float).values[:, None], df_test['nAdditions'].astype(float).values[:, None], df_test['nDeletions'].astype(float).values[:, None], df_test['Parents'].astype(float).values[:, None], df_test['nWords'].astype(float).values[:, None]))
p_train_1s,p_test_1s = RFCmodel(train_x_1s, df_train['Novelty '], test_x_1s, df_test['Novelty '])
p_train3_1s,p_test3_1s = RFCmodel(train_x_1s, df_train['Novelty3'], test_x_1s, df_test['Novelty3'])


# Create CV training and test scores for various training set sizes

