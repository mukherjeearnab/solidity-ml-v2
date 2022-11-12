from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

url = "../embeddings.csv"
data = pd.read_csv(url)

# data = data[:20000].iloc[:, 1:]
# data['Error_Label'] = (data['Error_Label'].str.contains('1')).astype(int)

print(data)

X, y = data.iloc[:, :64], data.iloc[:, 64:]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# scaler = StandardScaler()  # Fit on training set only.
# # Apply transform to both the training set and the test set.
# scaler.fit(X_train)


# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# Create an instance of SMOTE using the SMOTE() constructor, supplying the Random-State hyper-parameter.
sm = SMOTE(random_state=0)


# Applying SMOTE
X_train, y_train = sm.fit_resample(X_train, y_train)

with open('./X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)

with open('./X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f)

with open('./y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)

with open('./y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)
