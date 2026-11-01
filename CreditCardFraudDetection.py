import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble._forest import _parallel_build_trees
from tqdm.auto import tqdm
from joblib import parallel_backend
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
import xgboost as xg
from sklearn.linear_model import LogisticRegression


df=pd.read_csv(r"creditcard.csv")
df=df.drop('Time',axis=1)
print(df.isnull().sum())
df.head()


x=df.drop('Class',axis=1)
y=df['Class']
scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features=0.5,
    class_weight={0:1,1:10},
    n_jobs=-1
)


with tqdm(total=model.n_estimators, desc="Training Random Forest") as pbar:

    class ProgressCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = ProgressCallback

    try:
        model.fit(x_train, y_train)
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback

yhat = model.predict(x_test)

print(classification_report(y_test, yhat))

print(f'acc:{accuracy_score(yhat,y_test)}')
print(f'recall:{recall_score(yhat,y_test)}')
print(f'precision:{precision_score(y_test,yhat)}')

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=[0,1],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()



rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_dist = {
    'n_estimators': [200, 300],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 0.5],
    'class_weight': ['balanced', 'balanced_subsample']
}

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=5,
    scoring='recall',
    cv=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(x_train, y_train)

best_rf = search.best_estimator_

yhat = best_rf.predict(x_test)

print("Best Parameters:", search.best_params_)
print(classification_report(y_test, yhat))
