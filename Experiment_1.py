import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m = 100
X_raw = np.random.rand(m, 1) * 10
Y = 3 * X_raw + 4 + np.random.randn(m, 1) 

X = np.c_[np.ones((m, 1)), X_raw]


def solve_normal_equation(X, Y):
    X_T = X.T
    beta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(Y)
    return beta

beta_normal = solve_normal_equation(X, Y)
print("Parameters from Normal Equation [Intercept, Slope]:")
print(beta_normal.flatten())

def gradient_descent(X, Y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    beta = np.zeros((n, 1))
    loss_history = []
    
    for i in range(iterations):
        
        predictions = X.dot(beta)
        
        
        error = predictions - Y
        
        
        mse_loss = (1/m) * np.sum(error ** 2)
        loss_history.append(mse_loss)
        
        
        gradients = (2/m) * X.T.dot(error)
        
        
        beta = beta - learning_rate * gradients
        
    return beta, loss_history

beta_gd, losses = gradient_descent(X, Y, learning_rate=0.01, iterations=500)
print("\nParameters from Gradient Descent [Intercept, Slope]:")
print(beta_gd.flatten())


plt.figure(figsize=(8, 5))
plt.plot(losses, color='blue', linewidth=2)
plt.title('Loss (MSE) Reduction over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()