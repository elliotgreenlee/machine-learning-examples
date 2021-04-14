import pandas as pd
from stats import stats
from part1 import part1
from part2 import part2
from part_graduate import part_graduate


def main():
    """Main function"""

    """Read in data"""
    # Read in and clean up data
    df_messy = pd.read_csv("data/under5mortalityper1000.csv")
    df_clean = df_messy.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Remove missing entries
    df_data = df_clean.dropna(axis=0, how='any')

    # Separate country names and data
    df_countries = df_data['Country']
    df_deaths = df_data.drop('Country', axis=1)

    """Data familiarization"""
    stats(df_clean, df_deaths, df_countries)

    """Part 1"""
    part1(df_deaths, df_countries)

    """Part 2"""
    part2(df_deaths, df_countries)

    """Part Graduate"""
    part_graduate(df_deaths, df_countries)


if __name__ == "__main__":
    main()
