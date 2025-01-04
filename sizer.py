import pandas as pd
import numpy as np

"""
Module to balance the data set, looks for sample differences and samples down to a reasonable distribution of samples across feature column
"""



def create_balanced_sample(input_file, output_file, n_samples=500):
 
    df = pd.read_csv(input_file)
    
    lang_dist = df['language'].value_counts(normalize=True)
    
    # Initialize empty dataframe for results
    sampled_df = pd.DataFrame()
    
    for lang in lang_dist.index:
        # Calculate how many samples we want for this language
        n_lang_samples = int(np.round(n_samples * lang_dist[lang]))
        
        # Get all rows for this language
        lang_df = df[df['language'] == lang]
        
      
        sampled_lang_df = lang_df.sample(n=n_lang_samples, random_state=42)
            
        # Add to our result dataframe
        sampled_df = pd.concat([sampled_df, sampled_lang_df])
    
    # Shuffle the final dataframe
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print("\nOriginal Dataset Statistics:")
    print(f"Total examples: {len(df)}")
    print("\nLanguage distribution in original:")
    print(df['language'].value_counts())
    
    print("\nSampled Dataset Statistics:")
    print(f"Total examples: {len(sampled_df)}")
    print("\nLanguage distribution in sample:")
    print(sampled_df['language'].value_counts())
    
    # Save to CSV
    sampled_df.to_csv(output_file, index=False)
    print(f"\nSampled dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/filtered_questions.csv"
    output_file = "data/test_data.csv"
    create_balanced_sample(input_file, output_file, n_samples=500)