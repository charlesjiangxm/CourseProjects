from nn_forward import MLP
import torch
torch.set_printoptions(profile="short")  # or 'default'
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, linewidth=np.nan, precision=2, threshold=10000)
import cupy as cp
cp.cuda.Device(1).use()


def compute_mp_dist(m=10000, n=784, show_curve=True):
    """
    compute marchenko-pastur distribution
    """
    Q = m / n
    d_plus = 1 + 1/Q + 2*np.sqrt(1/Q)           # upper bound of eigen vlaue
    d_minus = 1 + 1/Q - 2*np.sqrt(1/Q)          # lower bound of eigen value
    d = np.linspace(d_minus, d_plus, 1000)      # d: assumed eigen values in [d-, d+]
    Prcm = (Q / (2 * np.pi)) * np.sqrt((d_plus - d) * (d - d_minus)) / d  # distribution of assumed eigen values

    if show_curve:
        plt.scatter(d, Prcm, color='r')
        plt.title("Density curve of Prcm(d)")
        plt.xlabel('eigenvalues with upper and lower bound')
        plt.ylabel('Density of eigen values')
        plt.show()

    return d, Prcm


def compute_eigen_iid(m=10000, n=784, solver='svd', show_curve=True):
    # create H(m,n) which has iid gussian for each entry.
    H = []
    for i in range(m * n):
        H.append(np.random.normal())
    H = cp.array(H).reshape(m, n)

    # create W from H, W is a semi-positive symmetric matrix
    W = (1/m)*H.T.dot(H)

    # compute eigen value of W
    eval_w = None
    evec_w = None
    vt_w = None
    s_w = None
    u_w = None
    if solver == "eigen":
        eval_w, evec_w = cp.linalg.eigh(W)
    else:
        vt_w, s_w, u_w = cp.linalg.svd(W)

    if show_curve:
        if solver == 'eigen':
            plt.hist(eval_w.real.get(), bins=50, color='b', density=True)
        else:
            plt.hist(s_w.get(), bins=50, color='b', density=True)
        plt.title("Density histogram of W matrix eigen value")
        plt.xlabel('W matrix eigenvalues')
        plt.ylabel('Eigen value density')
        plt.show()

    if solver == 'eigen':
        return eval_w, evec_w
    else:
        return vt_w, s_w, u_w


if __name__ == "__main__":
    # get the train, test set in matrix
    Mlp = MLP(model_loc="./model/mlp_model_1layer_256")
    mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label = Mlp.get_data_matrix()

    # compute empirical eigen density of Guassian noise matrix
    m, n = mnist_test_data.shape
    d, Prcm = compute_mp_dist(m, n)
    eval_w, evec_w = compute_eigen_iid(m, n, solver='eigen')

    plt.hist(eval_w.real.get(), bins=50, color='b', density=True)
    plt.scatter(d, Prcm, color='r')
    plt.title("Density histogram of W matrix eigen value")
    plt.xlabel('W matrix eigenvalues')
    plt.ylabel('Eigen value density')
    plt.show()
