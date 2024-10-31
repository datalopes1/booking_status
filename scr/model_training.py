import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def train_model(X, y, param_dict, preprocessor):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = lgb.LGBMClassifier(**param_dict, n_estimators=1000, random_state=42)
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test