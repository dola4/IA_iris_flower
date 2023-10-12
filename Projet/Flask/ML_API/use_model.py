import pickle
import numpy as np

#import classifier and scaler

local_classifier = pickle.load(open('classifier.pickle', 'rb'))
local_scaler = pickle.load(open('sc.pickle', 'rb'))


# New Data

new_data_1 = np.array([[20, 20000]])
new_predict_1 = local_classifier.predict(local_scaler.transform(new_data_1))
new_predict_proba_1 = local_classifier.predict_proba(local_scaler.transform(new_data_1))
print('Data : %s - Class : %s - Proba : %s'%(new_data_1, new_predict_1, new_predict_proba_1))

new_data_2 = np.array([[40, 20000]])
new_predict_2 = local_classifier.predict(local_scaler.transform(new_data_2))
new_predict_proba_2 = local_classifier.predict_proba(local_scaler.transform(new_data_2))
print('Data : %s - Class : %s - Proba : %s'%(new_data_2, new_predict_2, new_predict_proba_2))

new_data_3 = np.array([[42, 50000]])
new_predict_3 = local_classifier.predict(local_scaler.transform(new_data_3))
new_predict_proba_3 = local_classifier.predict_proba(local_scaler.transform(new_data_3))
print('Data : %s - Class : %s - Proba : %s'%(new_data_3, new_predict_3, new_predict_proba_3))
