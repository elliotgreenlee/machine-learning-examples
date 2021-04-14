import matplotlib.pyplot as plt
import pandas as pd


# Calculate statistics on the data, before using an machine learning techniques
def stats(df_clean, df_deaths, df_countries):
    # Get all country names
    df_all_countries = df_clean['Country']

    print("List of all countries:")
    print(df_all_countries)
    print("")

    # Get country names with missing data
    df_missing_countries = df_all_countries[~df_all_countries.index.isin(df_countries.index)]

    print("List of countries missing data:")
    print(df_missing_countries)
    print("")

    # Get stats from 1800
    average_1800 = df_deaths['1800'].mean()
    min_1800 = df_deaths['1800'].min()
    idxmin_1800 = df_deaths['1800'].idxmin()
    max_1800 = df_deaths['1800'].max()
    idxmax_1800 = df_deaths['1800'].idxmax()

    print("1800 Stats")
    print("\tAverage deaths: {}".format(average_1800))
    print("\tLowest deaths: {} in {}".format(min_1800, df_countries[idxmin_1800]))
    print("\tHighest deaths: {} in {}".format(max_1800, df_countries[idxmax_1800]))
    print("")

    # Get stats from 2015
    average_2015 = df_deaths['2015'].mean()
    min_2015 = df_deaths['2015'].min()
    idxmin_2015 = df_deaths['2015'].idxmin()
    max_2015 = df_deaths['2015'].max()
    idxmax_2015 = df_deaths['2015'].idxmax()

    print("2015 Stats")
    print("\tAverage deaths: {}".format(average_2015))
    print("\tLowest deaths: {} in {}".format(min_2015, df_countries[idxmin_2015]))
    print("\tHighest deaths: {} in {}".format(max_2015, df_countries[idxmax_2015]))
    print("")

    # Get stats on change from 2015 to 1800
    df_delta = df_deaths['2015'] - df_deaths['1800']
    print("Change Stats")
    print("Least Change")
    print("{}".format(df_delta.nlargest(5)))
    print("{}".format(df_countries[df_delta.nlargest(5).index.values.copy()]))
    print("")
    print("Most Change")
    print("{}".format(df_delta.nsmallest(5)))
    print("{}".format(df_countries[df_delta.nsmallest(5).index.values.copy()]))
    print("")
    print("Average Change")
    print(df_delta.mean())
    print("")

    # Plot averages from 1800 to 2015
    df_average = df_deaths.mean()
    print("Average deaths per year")
    print(df_average.values)

    plt.clf()
    plt.xlabel("Year")
    plt.ylabel("Deaths per 1000 Live Births")
    ax = df_average.plot()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/average_deaths_plot.png", facecolor='white')
