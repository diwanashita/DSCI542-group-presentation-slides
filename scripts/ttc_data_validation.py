# data_validation.py
# Tiffany Timber's Breast Cancer Predictor and some help of ChatGPT were used to to ensure correct syntax of this file
# Tiffany Timber's Breast Cancer Predictor: 
# https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/split_n_preprocess.py

# Imports
import os
import click
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

@click.command()
@click.option('--input-path', type=str, help="Path to the raw TTC bus delay data (CSV format)")
@click.option('--output-path', type=str, help="Path to save the cleaned TTC dataset (CSV format)")
def main(input_path: str, output_path: str):
    """Validates and cleans TTC bus delay data, and saves the cleaned dataset."""

    # Check if file exists and is a CSV
    if not os.path.exists(input_path) or not input_path.endswith('.csv'):
        raise FileNotFoundError("Error: File is either missing or not in CSV format.")

    try:
        # Load the CSV file and parse dates
        ttc = pd.read_csv(input_path)
        print("File loaded successfully!")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    # Schema validation
    schema = pa.DataFrameSchema(
        {
            "Route": pa.Column(str, nullable=True),
            "Day": pa.Column(str, checks=[
                pa.Check.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            ]),
            "Location": pa.Column(str),
            "Incident": pa.Column(str, checks=[
                pa.Check.isin([
                    "Cleaning - Unsanitary", "Collision - TTC", "Mechanical", "Operations - Operator",
                    "Diversion", "Emergency Services", "Utilized Off Route", "Investigation",
                    "Road Blocked - NON-TTC Collision", "Vision", "General Delay", "Security"
                ])
            ]),
            #"Min Delay": pa.Column(int, checks=[pa.Check.ge(0)]),
            #"Min Gap": pa.Column(int, checks=[pa.Check.ge(0)]),
            #]),
            "Month": pa.Column(int, checks=[pa.Check.ge(1), pa.Check.le(12)]),
            "Hour": pa.Column(int, checks=[pa.Check.ge(0), pa.Check.le(23)]),
        },
        checks=[
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found"),
            #pa.Check(lambda df: df["Min Delay"].isna().mean() <= 0.10, error="Min Delay missingness exceeds 10%"),
            #pa.Check(lambda df: df["Min Gap"].isna().mean() <= 0.10, error="Min Gap missingness exceeds 10%"),
        ]
    )

    try:
        schema.validate(ttc, lazy=True)
        print("Data validation passed!")
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"Data validation failed:\n{e}")

    # Save cleaned data
    # Used ChatGPT for this
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        raise NotADirectoryError(f"The path '{output_dir}' exists but is not a directory.")

    ttc.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == '__main__':
    main()