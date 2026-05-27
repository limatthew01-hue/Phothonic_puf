import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from puf_network import simulate_chip, get_response

N_STAGES = 8
N_ATTACK = 5000

target_chip = simulate_chip(n_stages=N_STAGES, seed=0)
X = np.random.uniform(0, 2*np.pi, size=(N_ATTACK, N_STAGES))
y = np.array([get_response(target_chip, c) for c in X])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM...")
svm = SVR(kernel="rbf", C=10, epsilon=0.01)
svm.fit(X_train, y_train)
svm_rmse = mean_squared_error(y_test, svm.predict(X_test))**0.5

print("Training MLP...")
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_rmse = mean_squared_error(y_test, mlp.predict(X_test))**0.5

baseline = np.std(y_test)
print(f"\nBaseline RMSE (random guess): {baseline:.4f}")
print(f"SVM attack RMSE:              {svm_rmse:.4f}")
print(f"MLP attack RMSE:              {mlp_rmse:.4f}")
print(f"\nSVM attack reduction:  {((baseline - svm_rmse)/baseline)*100:.1f}%")
print(f"MLP attack reduction:  {((baseline - mlp_rmse)/baseline)*100:.1f}%")

# Plot predictions vs actual
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test[:200], svm.predict(X_test)[:200], alpha=0.4, s=10, color="#378ADD")
plt.plot([0,1],[0,1], 'r--', linewidth=1)
plt.xlabel("Actual response")
plt.ylabel("Predicted response")
plt.title(f"SVM attack (RMSE={svm_rmse:.4f})")

plt.subplot(1, 2, 2)
plt.scatter(y_test[:200], mlp.predict(X_test)[:200], alpha=0.4, s=10, color="#1D9E75")
plt.plot([0,1],[0,1], 'r--', linewidth=1)
plt.xlabel("Actual response")
plt.ylabel("Predicted response")
plt.title(f"MLP attack (RMSE={mlp_rmse:.4f})")

plt.tight_layout()
plt.savefig("attack_results.png", dpi=150)
plt.show()