from scaling import *
from lwrmodel import *

#here we are tuning the bandwidth
#to find the best onle

bw = np.logspace(-2, 0, 10) #bandwidth values to try
best_bw = None
best_mse = float('inf')

for b in bw:
    model = local_weight_regression(bandwidth=b)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = np.mean((y_test - y_pred)**2)
    print(f'Bandwidth: {b:.4f}, MSE: {mse:.4f}')
    if mse < best_mse:
        best_mse = mse
        best_bw = b
print(f'Best Bandwidth: {best_bw:.4f}, Best MSE: {best_mse:.4f}')
