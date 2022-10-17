import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KernelCenterer
import os

np.random.seed(0)

os.makedirs('../week_2/data', exist_ok=True)


def save_data(X_train, X_test, y_train, y_test, type='linear'):
    np.savetxt(f'data/X_train_{type}.txt', X_train)
    np.savetxt(f'data/X_test_{type}.txt', X_test)
    np.savetxt(f'data/y_train_{type}.txt', y_train)
    np.savetxt(f'data/y_test_{type}.txt', y_test)

def generate_linear_data(n=200, p=10):
    b = np.random.normal()
    w = np.random.normal(size=(p, 1))
    X = np.random.normal(size=(n, p))
    noise = np.random.normal(scale=5, size=(n, 1))
    y = X @ w + b + noise
    y /= np.linalg.norm(y, axis=0)
    y -= y.mean()
    y[y > 0] = 1
    y[y < 0] = -1
    X += np.random.normal(scale=0.1, size=(1, p))
    return X, y


def generate_polynomial_data(n=200, p=10):
    b = np.random.normal()
    X = np.random.normal(size=(n, p))
    K = KernelCenterer().fit_transform(polynomial_kernel(X, degree=3))
    w = np.random.normal(size=(n, 1))
    noise = np.random.normal(scale=5, size=(n, 1))
    y = K @ w + b + noise
    y -= y.mean()
    y /= np.linalg.norm(y, axis=0)
    y[y > 0] = 1
    y[y < 0] = -1
    X += np.random.normal(scale=0.1, size=(1, p))
    return X, y


def main():
    X_train, X_test, y_train, y_test = train_test_split(*generate_linear_data(n=500, p=20), train_size=0.5)
    save_data(X_train, X_test, y_train, y_test, type='linear')
    X_train, X_test, y_train, y_test = train_test_split(*generate_polynomial_data(n=500, p=5), train_size=0.5)
    save_data(X_train, X_test, y_train, y_test, type='poly')


if __name__ == '__main__':
    main()
