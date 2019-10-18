"""
implementation of PCA algorithm and plot the result

"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA

figure_id = 1

def parse_file(filename):
    with open(filename, 'r') as f:
        content = f.readlines()

    labels = []
    diseases = []
    data = []

    for line in content:
        attributes = re.split('\t', line.strip())

        record = []
        for attr in attributes[:-1]:
            record.append(float(attr))
        data.append(record)

        if attributes[-1] not in diseases:
            diseases.append(attributes[-1])
        labels.append(diseases.index(attributes[-1]))

    return data, labels


def pca(data, labels):
    data = np.array(data)
    mean_vector = np.mean(data, axis=0)
    data_adjusted = data - mean_vector
    covariance = np.dot(np.transpose(data_adjusted), data_adjusted) / np.size(data, axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    top_indices = np.argpartition(eigenvalues, -2)[-2:]
    if eigenvalues[top_indices[0]] < eigenvalues[top_indices[1]]:
        top_indices[0], top_indices[1] = top_indices[1], top_indices[0]
    components = np.dot(data_adjusted, eigenvectors[:, top_indices])
    plot_result(components, labels, "PCA")


def tsne(data, labels):
    visual_model = TSNE(n_components=2, random_state=0)
    dxy = visual_model.fit_transform(data)
    plot_result(dxy, labels, "TSNE")


def svd(data, labels):
    visual_model = TruncatedSVD(n_components=2)
    dxy = visual_model.fit_transform(data)
    plot_result(dxy, labels, "SVD")


def plot_result(data, labels, title):
    global figure_id
    plt.figure(figure_id)
    figure_id += 1
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title(title)


if __name__ =='__main__':
    if len(sys.argv) < 2:
        filename = "pca_a.txt"
    else:
        filename = sys.argv[1]

    data, labels = parse_file(filename)

    pca(data, labels)
    tsne(data, labels)
    svd(data, labels)

    plt.show() # show figures all at once
    
