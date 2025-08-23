from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit

time_series_cv = TimeSeriesSplit(n_splits=3)

def get_logistic_model():
    model = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1, 10],
        cv=time_series_cv,
        penalty='l2',
        solver='saga',
        random_state=42,
        max_iter=10000,
        scoring='f1',
        n_jobs=-1
    )
    return model

def get_logistic_model_improved():
    model = LogisticRegressionCV(
        class_weight='balanced',
        Cs=[0.0001, 0.001, 0.001, 0.1, 1],
        cv=time_series_cv,
        penalty='elasticnet',
        l1_ratios=[0.2, 0.5, 0.7],
        solver='saga',
        random_state=42,
        max_iter=20000,
        scoring='f1',
        n_jobs=-1
    )
    return model