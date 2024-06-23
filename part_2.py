import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score

data = pd.read_csv('data.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm = SVC()
knn = KNeighborsClassifier()
nb = GaussianNB()


svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)


y_pred_svm = svm.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)


metrics = {
    'SVM': {
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Recall': recall_score(y_test, y_pred_svm, average='weighted'),
        'Precision': precision_score(y_test, y_pred_svm, average='weighted')
    },
    'KNN': {
        'Accuracy': accuracy_score(y_test, y_pred_knn),
        'Recall': recall_score(y_test, y_pred_knn, average='weighted'),
        'Precision': precision_score(y_test, y_pred_knn, average='weighted')
    },
    'Naive Bayes': {
        'Accuracy': accuracy_score(y_test, y_pred_nb),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'Precision': precision_score(y_test, y_pred_nb, average='weighted')
    }
}

for algo, metric in metrics.items():
    print(f"{algo} - Accuracy: {metric['Accuracy']:.2f}, Recall: {metric['Recall']:.2f}, Precision: {metric['Precision']:.2f}")
