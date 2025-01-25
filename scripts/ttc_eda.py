# ttc_eda.py
# Tiffany Timber's Breast Cancer Predictor and some help of ChatGPT were used to ensure correct syntax of this file
# Tiffany Timber's Breast Cancer Predictor: 
# https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/split_n_preprocess.py

# Imports
import os
import click
import pandas as pd
import altair as alt

@click.command()
@click.option('--input-path', type=str, help="Path to the cleaned TTC dataset (CSV format)")
@click.option('--output-dir', type=str, help="Directory to save EDA plots")
def main(input_path: str, output_dir: str):
    """Performs exploratory data analysis on the TTC bus delay dataset and saves visualizations."""

    # Check if file exists and is a CSV
    if not os.path.exists(input_path) or not input_path.endswith('.csv'):
        raise FileNotFoundError("Error: File is either missing or not in CSV format.")

    try:
        # Load the cleaned CSV file
        ttc = pd.read_csv(input_path)
        print("Cleaned dataset loaded successfully!")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # EDA Visualizations

    # 1. Incident Types Distribution
    incident_types_chart = (
        alt.Chart(ttc)
        .mark_bar()
        .encode(
            x=alt.X("Incident:N", title="Incident Type", sort="-y"),
            y=alt.Y("count()", title="Number of Incidents"),
            color=alt.Color("Incident:N", legend=None)
        )
        .properties(title="Distribution of Incident Types")
    )
    incident_types_chart.save(f"{output_dir}/incident_types.png")
    print("Saved: incident_types.png")

    # 2. Incidents by Hour
    incidents_by_hour = (
        alt.Chart(ttc)
        .mark_line(point=True)
        .encode(
            x=alt.X("Hour:O", title="Hour of Day"),
            y=alt.Y("count()", title="Number of Incidents"),
            color=alt.Color("Hour:O", legend=None)
        )
        .properties(title="Hourly Distribution of Incidents")
    )
    incidents_by_hour.save(f"{output_dir}/incidents_by_hour.png")
    print("Saved: incidents_by_hour.png")

    # 3. Incidents by Day of the Week
    delay_by_day = (
        alt.Chart(ttc)
        .mark_bar()
        .encode(
            x=alt.X("Day:N", title="Day of the Week", sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            y=alt.Y("count()", title="Number of Incidents"),
            color=alt.Color("Day:N", legend=None)
        )
        .properties(title="Distribution of Incidents by Day of the Week")
    )
    delay_by_day.save(f"{output_dir}/incidents_by_day.png")
    print("Saved: incidents_by_day.png")

    # 4. Incidents by Month
    incidents_by_month = (
        alt.Chart(ttc)
        .mark_bar()
        .encode(
            x=alt.X("Month:O", title="Month"),
            y=alt.Y("count()", title="Number of Incidents"),
            color=alt.Color("Month:O", legend=None)
        )
        .properties(title="Number of Incidents by Month")
    )
    incidents_by_month.save(f"{output_dir}/incidents_by_month.png")
    print("Saved: incidents_by_month.png")

    print("EDA completed. Visualizations saved!")

if __name__ == '__main__':
    main()
