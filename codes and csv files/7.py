
"""
7. Implement Principal Components Analysis
Write a function that implements PCA on a data frame. The function arguments should be
 a data frame and a boolean value indicated whether to whiten the data (True) or not 
 (False). The function should return the data means, the standard deviations (all 1s 
 if whitening is not used), the eigenvalues, the eigenvectors, and a new data frame 
 with the original data projected onto the eigenvectors. The algorithm outline is 
 below.

 Submitted by : Gmon Kuzhiyanikkal

"""


import numpy as np
import pandas as pd

def pca(data, normalize=True):
    # assign to A the data as a numpy matrix
    A = data.drop(['Y'], axis=1).to_numpy()

    # assign to m the mean values of the columns of A
    m = np.mean(A, axis=0)

    # assign to D the difference matrix A - m
    D = A - m

    # if normalize is true
    #    Compute the standard deviations of each column
    # else
    #    Assign all 1s to the standard deviation vector (1 for each column)
    if normalize:
        std_devs = np.std(D, axis=0)
    else:
        std_devs = np.ones(D.shape[1])

    # Divide each column by its standard deviation vector
    D = D / std_devs

    # assign to U, S, V the result of running np.svd on D, with full_matrices=False
    U, S, V = np.linalg.svd(D, full_matrices=False)

    # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
    #   divided by the degrees of freedom (N-1). The values are sorted.
    eigenvalues = (S ** 2) / (D.shape[0] - 1)

    # project the data onto the eigenvectors. Treat V as a transformation 
    #   matrix and right-multiply it by D transpose. The eigenvectors of A 
    #   are the rows of V. The eigenvectors match the order of the eigenvalues.
    projected_data = np.dot(D, V.T)

    # create a new data frame out of the projected data
    projected_data = pd.DataFrame(projected_data)

    # return the means, standard deviations, eigenvalues, eigenvectors, and projected data
    return m, std_devs, eigenvalues, V, projected_data,D


# load the csv file
data = pd.read_csv("pcatestdata.csv")
# call the pca function for non-whitened
m, std_devs, eigenvalues, eigenvectors, projected_data,D = pca(data, normalize=False)
print("--------non whitened resuls----------")
print('\n')
print("Means:", m)
print('\n')
print("D (no whitening):",D)
print("\n")
print("Standard deviations:", std_devs)
print('\n')
print("Eigenvalues:\n", eigenvalues)
print('\n')
print("Eigenvectors:\n", eigenvectors)
print('\n')
print("Projected data:\n", projected_data)
print('\n')
# call the pca function for whitened
m, std_devs, eigenvalues, eigenvectors, projected_data,D = pca(data, normalize=True)
print("--------whitened resuls----------")
print('\n')
print("Means:", m)
print("\n")
print("Data after whitening:", D)
print('\n')
print("Standard deviations:", std_devs)
print('\n')
print("Eigenvalues:\n", eigenvalues)
print('\n')
print("Eigenvectors:\n", eigenvectors)
print('\n')
print("Projected data:\n", projected_data)
print('\n')



