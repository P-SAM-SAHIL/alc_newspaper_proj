import pandas as pd
from datasets import load_dataset
import argparse
import os

def process_american_stories(year: str, metadata_csv_path: str, output_dir: str = "."):
    """
    Fetches AmericanStories data for a specific year, merges it with Chronicling America 
    metadata, and outputs a streamlined CSV.
    """
    print(f"--- Starting pipeline for year: {year} ---")
    
    # ---------------------------------------------------------
    # 1. Data Ingestion
    # ---------------------------------------------------------
    print(f"Loading article data from HuggingFace for {year}...")
    try:
        # Load the dataset for the specific year
        dataset = load_dataset(
            "dell-research-harvard/AmericanStories",
            "subset_years",
            year_list=[year],
            trust_remote_code=True
        )
        # Convert the specific year's dataset to a Pandas DataFrame
        df_articles = dataset[year].to_pandas()
        print(f"Successfully loaded {len(df_articles)} articles.")
    except Exception as e:
        print(f"Error loading data from HuggingFace: {e}")
        return

    print(f"Loading metadata from {metadata_csv_path}...")
    try:
        df_meta = pd.read_csv(metadata_csv_path)
        print(f"Successfully loaded {len(df_meta)} metadata records.")
    except Exception as e:
        print(f"Error loading metadata CSV: {e}")
        return

    # ---------------------------------------------------------
    # 2. Data Transformation & Cleaning
    # ---------------------------------------------------------
    print("Extracting LCCN from article_id...")
    # article_id format: '1_1870-01-01_p1_sn82014899_00211105483_1870010101_0773'
    # Splitting by '_' and grabbing the 4th element (index 3) gives the LCCN
    df_articles['LCCN'] = df_articles['article_id'].apply(
        lambda x: str(x).split('_')[3] if isinstance(x, str) and len(str(x).split('_')) > 3 else None
    )

    print("Cleaning and standardizing LCCN columns...")
    # Strip whitespace and ensure string type for a perfect 1:1 match
    df_articles['LCCN'] = df_articles['LCCN'].astype(str).str.strip()
    df_meta['LCCN'] = df_meta['LCCN'].astype(str).str.strip()

    # ---------------------------------------------------------
    # 3. The Merge (Inner Join)
    # ---------------------------------------------------------
    print("Performing Inner Join on LCCN...")
    df_merged = pd.merge(df_articles, df_meta, on='LCCN', how='inner')
    print(f"Merge complete. Resulting dataset has {len(df_merged)} rows.")

    # ---------------------------------------------------------
    # 4. Final Output Formatting
    # ---------------------------------------------------------
    print("Streamlining columns...")
    # Define the 9 essential analytical columns requested
    columns_to_keep = [
        'article_id', 'date', 'headline', 'article', 'newspaper_name', 
        'LCCN', 'State', 'County', 'City'
    ]
    
    # Ensure all requested columns exist before filtering to avoid KeyError
    missing_cols = [col for col in columns_to_keep if col not in df_merged.columns]
    if missing_cols:
        print(f"Warning: The following expected columns are missing in the merged data: {missing_cols}")
        # Keep only the columns that actually exist
        columns_to_keep = [col for col in columns_to_keep if col in df_merged.columns]

    df_final = df_merged[columns_to_keep]

    # Save to CSV
    output_filename = f"american_stories_{year}_merged.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    print(f"Saving final dataset to {output_filepath}...")
    df_final.to_csv(output_filepath, index=False)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Setup command-line arguments for easy execution
    parser = argparse.ArgumentParser(description="Merge AmericanStories dataset with Chronicling America metadata.")
    parser.add_argument("--year", type=str, required=True, help="The year to process (e.g., '1963')")
    parser.add_argument("--meta", type=str, required=True, help="Path to the Master Metadata CSV file")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save the merged CSV (defaults to current directory)")
    
    args = parser.parse_args()
    
    process_american_stories(year=args.year, metadata_csv_path=args.meta, output_dir=args.outdir)