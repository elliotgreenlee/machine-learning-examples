import matplotlib.pyplot as plt
import numpy as np
from ml_functions import svd_reduce, dunn_index, em_gaussian


def part_graduate(df_deaths, df_countries):
    # Reduce to k=5 principal components
    df_principle_components = svd_reduce(df_deaths, 5, df_countries)

    # Reduce to k=2 principal components
    df_graphing_components = svd_reduce(df_deaths, 2, df_countries)

    # 2 principal components
    print("k vs dunn index for 2 principal components with expectation maximization")
    for k in range(2, 10):
        means, labels = em_gaussian(df_graphing_components, k)

        clusters = [[] for i in range(k)]
        for i, (label, country) in enumerate(zip(labels, df_graphing_components.as_matrix())):
            clusters[int(label)].append(country)

        print(k, dunn_index(means, clusters))
    print("")

    k = 2
    means, labels = em_gaussian(df_graphing_components, k)

    graph_clusters = np.zeros((k, len(labels), len(df_graphing_components.iloc[0])))
    for i, (label, country) in enumerate(zip(labels, df_graphing_components.as_matrix())):
        graph_clusters[int(label)][i] = country

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    for i in range(k):
        plt.scatter(graph_clusters[i, :, 0], graph_clusters[i, :, 1], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    # plt.show()
    plt.savefig("results/graphs/em_2pc_data_2_cluster_plot.png", facecolor='white')

    k = 3
    means, labels = em_gaussian(df_graphing_components, k)

    graph_clusters = np.zeros((k, len(labels), len(df_graphing_components.iloc[0])))
    for i, (label, country) in enumerate(zip(labels, df_graphing_components.as_matrix())):
        graph_clusters[int(label)][i] = country

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    for i in range(k):
        plt.scatter(graph_clusters[i, :, 0], graph_clusters[i, :, 1], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    # plt.show()
    plt.savefig("results/graphs/em_2pc_data_3_cluster_plot.png", facecolor='white')

    # 5 principal components
    print("k vs dunn index for 5 principal components with expectation maximization")
    for k in range(2, 10):
        means, labels = em_gaussian(df_principle_components, k)

        clusters = [[] for i in range(k)]
        for i, (label, country) in enumerate(zip(labels, df_principle_components.as_matrix())):
            clusters[int(label)].append(country)

        print(k, dunn_index(means, clusters))
    print("")

    k = 2
    means, labels = em_gaussian(df_principle_components, k)

    graph_clusters = np.zeros((k, len(labels), len(df_graphing_components.iloc[0])))
    for i, (label, country) in enumerate(zip(labels, df_graphing_components.as_matrix())):
        graph_clusters[int(label)][i] = country

    plt.clf()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.style.use('fivethirtyeight')

    for i in range(k):
        plt.scatter(graph_clusters[i, :, 0], graph_clusters[i, :, 1], s=6)

    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    # plt.show()
    plt.savefig("results/graphs/em_5pc_data_2_cluster_plot.png", facecolor='white')
