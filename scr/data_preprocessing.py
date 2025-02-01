import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from category_encoders import TargetEncoder, OrdinalEncoder

def get_preprocessor():
    """
    Create the pre-processing pipeline

    Returns:
        ColumnTransformer: A object containing all the transformations applied to the data.
    """
    cat_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 
                    'repeated_guest', 'required_car_parking_space']
    
    num_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                    'lead_time', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                    'avg_price_per_room', 'no_of_special_requests']
    
    ordinal_features = ['arrival_date', 'arrival_month']

    cat_transformer = Pipeline([
        ('imput_cat', CategoricalImputer(imputation_method='frequent')),
        ('encoder_cat', TargetEncoder())
    ])

    num_transformer = Pipeline([
        ('imput_num', MeanMedianImputer(imputation_method='median'))
    ])

    ordinal_transformer = Pipeline([
        ('imput_or', MeanMedianImputer(imputation_method='median')),
        ('encoder_or', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_features),
            ('num', num_transformer, num_features),
            ('ordinal', ordinal_transformer, ordinal_features)
        ]
    )
    
    return preprocessor