"""
Elliot Greenlee
528 Project 3
November 5, 2017

Print statistics about a pandas dataframe
"""


def stats(df_data):
    print("Data:")
    print(df_data)
    print("")

    print("Minimum:")
    print(df_data.min(axis=0))
    print("")

    print("Maximum:")
    print(df_data.max(axis=0))
    print("")

    print("Mean:")
    print(df_data.mean(axis=0))
    print("")

    print("Standard Deviation:")
    print(df_data.std(axis=0))
    print("")
