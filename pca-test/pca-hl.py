import pickle
from sklearn.decomposition import PCA  # Make an instance of the Model

with open('./X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('./X_test.pickle', 'rb') as f:
    X_test = pickle.load(f)


pca = PCA(0.5)

pca.fit(X_train)

X_train_t = pca.transform(X_train)
X_test_t = pca.transform(X_test)

print(len(X_train_t[0]))

with open('./X_train_t.pickle', 'wb') as f:
    pickle.dump(X_train_t, f)

with open('./X_test_t.pickle', 'wb') as f:
    pickle.dump(X_test_t, f)
