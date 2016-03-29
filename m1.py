import numpy as np
import pandas as pd
import sklearn.semi_supervised as ss
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
import sklearn.ensemble as ske
import sklearn.decomposition as deco
import sklearn.linear_model as lm
import sklearn.svm as svm

def classification_score(y_pred,y_true):
    score = skmet.accuracy_score(y_true,y_pred)
    print score
    return score

def extra_class(X,X_val,y):
    print 'extra tree classification'
    ## extra trees classifier
    scorefun = skmet.make_scorer(classification_score)
    extra = ske.ExtraTreesClassifier()

    param_grid = {'n_estimators':[300],'criterion':['gini']}
    grid_search = skgs.GridSearchCV(extra,param_grid,scoring=scorefun,cv=2)
    grid_search.fit(X,y)
    score_y = grid_search.best_score_
    best_y = grid_search.best_estimator_
    print score_y
    print best_y
    print 'predict probabilities'
    y_pred = best_y.predict_proba(X_val)
    return y_pred

def sgd_class(X,X_val,y):
    print 'sgd classification'
    ## sgd classification
    scorefun = skmet.make_scorer(classification_score)
    sgd = lm.SGDClassifier()

    param_grid = {'loss':['log','modified_huber']}
    grid_search = skgs.GridSearchCV(sgd,param_grid,scoring=scorefun,cv=2)
    grid_search.fit(X,y)
    score_y = grid_search.best_score_
    best_y = grid_search.best_estimator_
    print score_y
    print best_y
    print 'predict probs'
    y_pred = best_y.predict_proba(X_val)
    return y_pred

def svc_class(X,X_val,y):
    print 'svc classification'
    ## svc classification
    scorefun = skmet.make_scorer(classification_score)
    svc = svm.NuSVC()
    
    param_grid = {'kernel':['rbf','sigmoid']}
    grid_search = skgs.GridSearchCV(svc,param_grid,scoring=scorefun,cv=2)
    grid_search.fit(X,y)
    score_y = grid_search.best_score_
    best_y = grid_search.best_estimator_
    print score_y
    print best_y
    print 'predict probs'
    y_pred = best_y.predict_proba(X_val)
    return y_pred


## read in data
print 'read in data'
X = pd.read_csv('train.csv',sep=',',header=None)
X_val = pd.read_csv('validate.csv',sep=',',header=None)
y = pd.read_csv('train_y.csv',sep=',',header=None)

X = X.as_matrix()
y = y.as_matrix()
y = np.reshape(y,len(y))
X_val = X_val.as_matrix()
print 'successful read in data'

print 'pca'
pca = deco.PCA(n_components=50)
X_pca = pca.fit_transform(X)
X_val_pca = pca.transform(X_val)

print 'pca finished'

## Label Propagation
lpm = ss.LabelPropagation(gamma=0.3,max_iter=9)
#lpm = ss.LabelSpreading(kernel='knn',n_neighbors=20,max_iter=7,tol=0.02)
print 'fit model'
lpm.fit(X_pca,y)
print 'model fitted'
y = lpm.transduction_





#print 'predict'
y_pred = lpm.predict_proba(X_val_pca)
#print len(y_pred)
#print 'predicted ... write file'
##########################
### extra trees
#y_pred = extra_class(X,X_val,y)
y_pred = pd.DataFrame(data=y_pred)
print len(y_pred)
y_pred.to_csv('lpm2.csv',index=False,header=False)
print 'finished extra trees'
#############################
##### sgd
##y_pred = sgd_class(X,X_val,y)
##y_pred = pd.DataFrame(data=y_pred)
##print len(y_pred)
##y_pred.to_csv('sgd1.csv',index=False,header=False)
##print 'finished sgd'

###########################
### svc
##y_pred = svc_class(X,X_val,y)
##y_pred = pd.DataFrame(data=y_pred)
##print len(y_pred)
##y_pred.to_csv('svc1.csv',index=False,header=False)
##print 'finished svc'



