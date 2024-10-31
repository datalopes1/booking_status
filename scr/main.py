import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils import load_data
from preprocessing import create_preprocessor
from model_training import train_model

def main():

    data = load_data("../data/processed/clean_data.csv")

    features = data.drop(columns=['booking_status', 'arrival_year','arrival_year_month'], axis=1).columns.to_list()
    target = 'booking_status'

    X = data[features]
    y = data[target]

    cat_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'repeated_guest', 'required_car_parking_space']
    num_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'lead_time', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
    ordinal_features = ['arrival_date', 'arrival_month']

    preprocessor = create_preprocessor(cat_features, num_features, ordinal_features)

    param_dict =  {'learning_rate': 0.00681644874893244, 
                   'num_leaves': 382, 
                   'subsample': 0.9945007005142789, 
                   'colsample_bytree': 0.8447669219822008, 
                   'min_data_in_leaf': 1}
    
    clf, X_test, y_test = train_model(X, y, param_dict, preprocessor)
    model_series = pd.Series({'model': clf, 'features': features})
    model_series.to_pickle("../models/classifier.pkl")
    print("Model file saved in 'models/classifier.pkl'")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    metrics = {
        'Accuracy': float(round(accuracy, 4)),
        'F1 Score': float(round(f1, 4)),
        'ROC AUC Score': float(round(roc, 4))
    }
    
    return print(metrics)

if __name__ == "__main__":
    main()