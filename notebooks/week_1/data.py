import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.model_selection import train_test_split
import os

np.random.seed(0)

os.makedirs('data', exist_ok=True)


def save_data(X_train, X_test, y_train, y_test, type='linear'):
    np.savetxt(f'data/X_train_{type}.txt', X_train)
    np.savetxt(f'data/X_test_{type}.txt', X_test)
    np.savetxt(f'data/y_train_{type}.txt', y_train)
    np.savetxt(f'data/y_test_{type}.txt', y_test)


def generate_linear_data(n=200, p=10):
    w = np.random.normal(size=(p, 1))
    X = np.random.normal(size=(n, p))
    noise = np.random.normal(scale=1, size=(n, 1))
    y = X @ w + noise
    y /= np.linalg.norm(y, axis=0)
    return X, y


def generate_polynomial_data(n=200, p=10):
    X = np.random.normal(size=(n, p))
    K = polynomial_kernel(X, degree=3)
    w = np.random.normal(size=(n, 1))
    noise = np.random.normal(scale=10, size=(n, p))
    y = K @ w + noise
    y /= np.linalg.norm(y, axis=0)
    return X, y


def generate_nonlinear_data(n=200, p=10):
    r = np.random.normal(size=(n, p))
    theta = np.random.normal(size=(n, p))
    X = np.hstack((r * np.cos(theta), r * np.sin(theta)))
    w = np.random.normal(size=(p, 1))
    noise = np.random.normal(scale=10, size=(n, p))
    y = r @ w + noise
    y /= np.linalg.norm(y, axis=0)
    return X, y


def main():
    X_train, X_test, y_train, y_test = train_test_split(*generate_linear_data(n=200, p=20), train_size=0.5)
    save_data(X_train, X_test, y_train, y_test, type='linear')
    X_train, X_test, y_train, y_test = train_test_split(*generate_polynomial_data(n=400, p=5))
    save_data(X_train, X_test, y_train, y_test, type='poly')
    X_train, X_test, y_train, y_test = train_test_split(*generate_nonlinear_data(n=400, p=5))
    save_data(X_train, X_test, y_train, y_test, type='nonlinear')


if __name__ == '__main__':
    main()
