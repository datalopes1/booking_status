import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from category_encoders import TargetEncoder, OrdinalEncoder

def create_preprocessor(cat_features, num_features, ordinal_features):
    
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
