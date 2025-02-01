import os
import pandas as pd 
import joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def load_model(model_path):
    """
    Load a trained model

    Args:
        model_path(str): path to the model.

    Returns:
        sklearn.pipeline.Pipeline: pipeline containing the model and pre-processor.
    """
    return joblib.load(model_path)

def make_predictions(model, data):
    """
    Make predictions using the model

    Args:
        model (Pipeline): pipeline with the pre-processing and model.
        data (pd.DataFrame): features

    Returns:
        tuple:
            - predictions
            - probabilities
    """
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:,1]
    return predictions, probabilities

if __name__ == "__main__":

    data = pd.read_csv("data/processed/test_data.csv")
    results = pd.read_csv("data/processed/test_results.csv")
    model = load_model("models/classifier.pkl")

    predictions, probabilities = make_predictions(model, data)
    data['pred'] = predictions
    data['prob'] = probabilities.round(4)

    result_df = pd.concat([data, results], axis = 1)
    result_df = result_df[['booking_status', 'pred', 'prob']]
    print(result_df.head())

    result_df.to_excel("data/processed/predictions.xlsx", index = False)
    print("Predictions saved in data/processed/")



