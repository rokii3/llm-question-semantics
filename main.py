import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def prepare_multilabel_encoding(df):
    """
    Prepares multi-label encodings for 'feature1' and 'feature2' columns.

    Args:
        df: Pandas DataFrame with 'feature1' and 'feature2' columns containing lists of strings.

    Returns:
        A Pandas DataFrame with additional columns for multi-label encoding.
    """

    # Make sure feature columns are lists
    df['feature1'] = df['feature1'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['feature2'] = df['feature2'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Feature 1 Encoding:
    mlb_feature1 = MultiLabelBinarizer()
    encoded_feature1 = mlb_feature1.fit_transform(df['feature1'])
    encoded_feature1_df = pd.DataFrame(encoded_feature1, columns=[f"f1_{label}" for label in mlb_feature1.classes_])
    df = pd.concat([df.reset_index(drop=True), encoded_feature1_df.reset_index(drop=True)], axis=1)

    # Feature 2 Encoding:
    mlb_feature2 = MultiLabelBinarizer()
    encoded_feature2 = mlb_feature2.fit_transform(df['feature2'])
    encoded_feature2_df = pd.DataFrame(encoded_feature2, columns=[f"f2_{label}" for label in mlb_feature2.classes_])
    df = pd.concat([df.reset_index(drop=True), encoded_feature2_df.reset_index(drop=True)], axis=1)

    return df

if __name__ == '__main__':
    # Load the test data
    test_data = pd.read_csv("data/test_data.csv")

    # Apply multi-label encoding
    encoded_test_data = prepare_multilabel_encoding(test_data.copy())

    # Save the encoded data to a CSV file
    output_csv_path = "data/encoded_test_data.csv" # Change this if you want to save it somewhere else
    encoded_test_data.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Encoded data saved to: {output_csv_path}")


    # Print the result
    print(encoded_test_data.head())

    # Print columns to show that the new columns are there
    print("\nNew Columns:")
    print(encoded_test_data.columns)

    # Print the number of columns
    print("\nNumber of Columns:")
    print(len(encoded_test_data.columns))

    # Show the value counts for a few selected columns
    print("\nValue counts for f1_modality:")
    print(encoded_test_data['f1_modality'].value_counts())
    print("\nValue counts for f2_polar:")
    print(encoded_test_data['f2_polar'].value_counts())