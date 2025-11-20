from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def create_train_test_split(data, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): The entire dataset.
        target_column (str): The name of the column to be used as the target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing training and testing data splits (X_train, X_test, y_train, y_test).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def bootstrap_sample(data):
    """
    Creates a bootstrap sample of the data.

    A bootstrap sample is a random sample of the data taken with replacement.

    Args:
        data (pd.DataFrame): The dataset to sample from.

    Returns:
        pd.DataFrame: A bootstrap sample of the original data.
    """
    return resample(data)