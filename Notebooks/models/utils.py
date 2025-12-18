from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pandas as pd

def return_scores(estimator, X_train, X_test, y_train, y_test):
    estimator = estimator
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)

    train_recall = recall_score(y_train, estimator.predict(X_train))
    test_recall = recall_score(y_test, estimator.predict(X_test))

    df = pd.DataFrame({
        'Train Scores': [train_score, train_recall],
        'Test Scores': [test_score, test_recall]
    }, index=['Accuracy', 'Recall'])

    return df