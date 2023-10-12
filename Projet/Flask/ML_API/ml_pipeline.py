import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # data split
from sklearn.preprocessing import StandardScaler # data transform
from sklearn.linear_model import LogisticRegression # model
from sklearn.metrics import confusion_matrix, accuracy_score # matrice de confusion

# load dataset

filename = 'storepurchasedata.csv' # trouver le fichier
training_data = pd.read_csv(filename) # load de fichier


#describe dataset

description = training_data.describe()
#print(description)


#split input and output component

array = training_data.values
x = array[ : , 0: -1]
y = array[ : , -1]
test_proportion = .20
seed = 7


# data split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_proportion, random_state=seed) 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# define and train model

classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, Y_train)

# Evaliate performance

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
cm = confusion_matrix(Y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
print('Accuracy : ', accuracy_score(Y_test, y_pred))

# New Data

new_data_1 = np.array([[20, 20000]])
new_predict_1 = classifier.predict(sc.transform(new_data_1))
new_predict_proba_1 = classifier.predict_proba(sc.transform(new_data_1))
print('Data : %s - Accuracy : %s - Proba : %s'%(new_data_1, new_predict_1, new_predict_proba_1))

new_data_2 = np.array([[40, 20000]])
new_predict_2 = classifier.predict(sc.transform(new_data_2))
new_predict_proba_2 = classifier.predict_proba(sc.transform(new_data_2))
print('Data : %s - Accuracy : %s - Proba : %s'%(new_data_2, new_predict_2, new_predict_proba_2))

new_data_3 = np.array([[42, 50000]])
new_predict_3 = classifier.predict(sc.transform(new_data_3))
new_predict_proba_3 = classifier.predict_proba(sc.transform(new_data_3))
print('Data : %s - Accuracy : %s - Proba : %s'%(new_data_3, new_predict_3, new_predict_proba_3))

# save model

import pickle
model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

# save transform

scaler_file = 'sc.pickle'
pickle.dump(sc, open(scaler_file, 'wb'))


