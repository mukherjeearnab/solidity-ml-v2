from sklearn.metrics import f1_score, confusion_matrix
# from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle

with open('./X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('./X_test.pickle', 'rb') as f:
    X_test = pickle.load(f)

with open('./y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open('./y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)


# model = XGBClassifier(n_estimators=2, max_depth=2,
#                       learning_rate=1, objective='binary:logistic')

model = AdaBoostClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
f1_mac = f1_score(y_test, y_predict, average='macro')
f1_mic = f1_score(y_test, y_predict, average='micro')
f1_wei = f1_score(y_test, y_predict, average='weighted')

print(cm)
print(f1_mac)
print(f1_mic)
print(f1_wei)
