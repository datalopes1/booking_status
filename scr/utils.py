import pandas as pd

def load_data(file_path):
    """
    Load a .csv dataset and transform the target in bool dtype.
    
    - 'Not_Canceled' -> 0
    - 'Canceled' -> 1

    Args:
        file_path (str): Path to the .csv file. 

    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(file_path)
    df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
    return df
    
