from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat

data = loadmat('cardio.mat')
X = data['X']
y = data['y'].flatten()

y = (2 * y - 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)

param_grid = {
    "oneclasssvm__kernel": ["rbf", "linear", "poly"],
    "oneclasssvm__gamma": [0.001, 0.01, 0.1, 1],
    "oneclasssvm__nu": [0.01, 0.1, 0.5, 0.9]
}

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("oneclasssvm", OneClassSVM())
])

grid_search = GridSearchCV(pipeline, param_grid, scoring="balanced_accuracy", cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

y_pred_test = (y_pred_test + 1) // 2
y_test_pyod = (y_test + 1) // 2

balanced_acc = balanced_accuracy_score(y_test_pyod, y_pred_test)
print("Balanced Accuracy on Test Set:", balanced_acc)