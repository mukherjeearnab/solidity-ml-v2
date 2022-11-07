from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
import pickle

with open('./X_train_t.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('./X_test_t.pickle', 'rb') as f:
    X_test = pickle.load(f)

with open('./y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open('./y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)


model = XGBClassifier()

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
