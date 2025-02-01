import pandas as pd
import joblib
import lightgbm as lgb
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from data_preprocessing import get_preprocessor
from utils import load_data

def split_data(X, y, test_size= 0.20, random_state = 42):
    """
    Split data into test and training sets

    Args:
        X (pd.DataFrame): features 
        y (pd.Series): target
        test_size (float): training set proportion
        random_state(int): random seed

    Returns:
        tuple: train and test sets (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the LightGBM model 

    Args:
        X_train (pd.DataFrame): training set of features.
        y_train (pd.Series): training set of the target.

    Returns:
        lgb.LGBMClassifer: trained model.
    """
    params = {'learning_rate': 0.02780638516107387, 
              'num_leaves': 423, 
              'subsample': 0.8869411153014691, 
              'colsample_bytree': 0.6957945784046679, 
              'min_data_in_leaf': 10, 
              'verbose': -1}
    
    model = lgb.LGBMClassifier(**params, n_estimators = 1000, random_state = 42)
    
    with tqdm(total=model.n_estimators, desc="Training Progress") as pbar:
        def callback(env):
            pbar.update(1)
        
        model.fit(X_train, y_train, callbacks=[callback])
    
    print("\nThe model is ready")

    return model

def save_model(model, preprocessor, path):
    """
    Save the model and pre-processing pipeline. 

    Args:
        model (lgb.LGBMClassifier): trained model.
        preprocessor (ColumnTransformer): pre-processing pipeline.
        path (str): path to save the model. 

    Returns:
        None 
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    joblib.dump(pipeline, path)
    print(f"Model and preprocessor saved to {path}")

if __name__ == "__main__":
    data = load_data("data/raw/Hotel Reservations.csv")
    features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
    target = 'booking_status'
    
    preprocessor = get_preprocessor()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = split_data(X, y)

    train_data = pd.concat([X_train, y_train], axis = 1)
    train_data.to_csv("data/processed/train_data.csv", index = False)
    X_test.to_csv("data/processed/test_data.csv", index = False)
    y_test.to_csv("data/processed/test_results.csv", index = False)

    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_train_transformed = pd.DataFrame(X_train_transformed)

    X_train, X_test, y_train, y_test = split_data(X_train_transformed, y_train)
    model = train_model(X_train, y_train)
    
    save_model(model, preprocessor, "models/classifier.pkl")



