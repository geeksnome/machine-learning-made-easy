import pandas as pd 
msg=pd.read_csv('naivetext1.csv',names=['message','label'])
print('The dimension of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
print(X)
print(y)

#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
print(xtest.shape)
print(xtrain.shape)
print(ytest.shape)
print(ytrain.shape)

#output of count vectorises is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
xtrain_dtm=count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
print(count_vect.get_feature_names())
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df)
print(xtrain_dtm)

#Training Navie Bayes(NB) classifier on training data
from sklearn.naive_bayes import MultinomialNB
df=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=df.predict(xtest_dtm)

#printing accuarcy metrics
from sklearn import metrics
print('Accuracy Metrics')
print('Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precision')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))
