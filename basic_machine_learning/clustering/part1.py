import matplotlib.pyplot as plt
import numpy as np

from ml_functions import svd_reduce


def part1(df_deaths, df_countries):
    """Part 1"""
    # Normalize data
    # df_deaths_normalized = (df_deaths - df_deaths.mean()) / df_deaths.std()
    df_deaths_normalized = df_deaths

    # Perform SVD
    np_deaths_normalized = df_deaths_normalized.as_matrix()
    np_u, np_singular_values, np_vt = np.linalg.svd(np_deaths_normalized)

    # Decide a good choice of k
    k = 5

    # Graph Scree plot of singular values
    plt.clf()
    plt.xlabel("Factor Number k")
    plt.ylabel("Singular Value")
    plt.style.use('fivethirtyeight')
    plt.scatter(range(1, len(np_singular_values)+1), np_singular_values, s=6)
    plt.scatter(k, np_singular_values[k-1], s=8)
    ax = plt.gca()
    ax.annotate("  {}".format(k), (k, np_singular_values[k-1]))
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/scree_plot.png", facecolor='white')

    # Graph the percent of variance covered by the first k singular values vs k
    np_singular_values_squared = np.square(np_singular_values)
    np_percent_variances = np_singular_values_squared / np_singular_values_squared.sum()
    for i in range(1, len(np_percent_variances)):
        np_percent_variances[i] = np_percent_variances[i-1] + np_percent_variances[i]

    plt.clf()
    plt.xlabel("Factor Number k")
    plt.ylabel("Percentage of Variance up to k")
    plt.style.use('fivethirtyeight')
    plt.scatter(range(1, len(np_percent_variances)+1), np_percent_variances, s=6)
    plt.scatter(k, np_percent_variances[k-1], s=8)
    ax = plt.gca()
    ax.annotate("  {}".format(k), (k, np_percent_variances[k-1]))
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/percent_variance_plot.png", facecolor='white')

    # Reduce to k=5 principal components
    df_principle_components = svd_reduce(df_deaths, k, df_countries)

    # Reduce to k=2 principal components
    df_graphing_components = svd_reduce(df_deaths, 2, df_countries)

    # Graph the first two principal components
    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')
    plt.scatter(df_graphing_components['0'], df_graphing_components['1'], s=6)
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/graphs/two_principal_components_plot.png", facecolor='white')

    # Find some countries
    print("Principle Component 1")
    print("Largest 5:")
    print(df_principle_components.nlargest(5, '0'))
    print("Smallest 5:")
    print(df_principle_components.nsmallest(5, '0'))

    print("Principle Component 2")
    print("Largest 5:")
    print(df_principle_components.nlargest(5, '1'))
    print("Smallest 5:")
    print(df_principle_components.nsmallest(5, '1'))
