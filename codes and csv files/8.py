
"""
9.Apply PCA
Apply PCA to your data set. How many significant dimensions does it have? Which variables 
are correlated or uncorrelated? For your data, does it makes sense to use whitening?
What happens if you do not use whitening?

submitted by : Gmon Kuzhiyanikkal
"""



import numpy as np
import pandas as pd

def pca(data, normalize=True):
    # assign to A the data as a numpy matrix
    A = data.to_numpy()

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
    return m, std_devs, eigenvalues, V, projected_data


#load the csv file
data = pd.read_csv("insurance.csv")
# call the pca function for non-whitened
m, std, eigenvalues, eigenvectors, projected_data = pca(data, normalize=False)

# Determine the number of significant dimensions
print("\n")
print("-----------------non-whiten------------------")
num_sig_dims = np.sum(eigenvalues > 1)
print("Number of significant dimensions:", num_sig_dims)
print("\n")
# Analyze correlation between variables
corr_matrix = np.corrcoef(projected_data.T)
print("Correlation matrix:\n", corr_matrix)
print("\n")


data = pd.read_csv("insurance.csv")
# call the pca function for whitened
m, std, eigenvalues, eigenvectors, projected_data = pca(data, normalize=True)

# Determine the number of significant dimensions
print("\n")
print("-----------------whiten------------------")
num_sig_dims = np.sum(eigenvalues > 1)
print("Number of significant dimensions:", num_sig_dims)
print("\n")
# Analyze correlation between variables
corr_matrix = np.corrcoef(projected_data.T)
print("Correlation matrix:\n", corr_matrix)
print("\n")
