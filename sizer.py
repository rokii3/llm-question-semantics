import pandas as pd
import numpy as np

"""
Module to balance the data set, looks for sample differences and samples down to a reasonable distribution of samples across feature column.

balances the dataset by undersampling each language to have the same number of samples as the language with the fewest samples, and then saves the balanced dataset to a new CSV file

"""

def create_balanced_sample(input_file, output_file):
    df = pd.read_csv(input_file)
    min_samples = df['language'].value_counts().min()
    # Initialize empty dataframe for results
    sampled_df = pd.DataFrame()
    
    for lang in df['language'].unique(): # Calculate how many samples we want for this language
        lang_df = df[df['language'] == lang] # Get all rows for this language
        sampled_lang_df = lang_df.sample(n=min_samples, random_state=42)
        sampled_df = pd.concat([sampled_df, sampled_lang_df]) # Add to our result dataframe
    
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the final dataframe
    
    print("\nOriginal Dataset Statistics:")
    print(f"Total examples: {len(df)}")
    print("\nLanguage distribution in original:")
    print(df['language'].value_counts())
    
    print("\nSampled Dataset Statistics:")
    print(f"Total examples: {len(sampled_df)}")
    print("\nLanguage distribution in sample:")
    print(sampled_df['language'].value_counts())
    
    sampled_df.to_csv(output_file, index=False)
    print(f"\nSampled dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = "sl-project-kokot\qtytp-all-encoded.csv"
    output_file = "sl-project-kokot\qtytp-balanced.csv"
    create_balanced_sample(input_file, output_file)