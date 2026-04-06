import notebooks.nets as nets
import numpy as np
import matplotlib.pyplot as plt

# Funcion no lineal

N = 100
x1 = np.linspace(-np.pi, np.pi, N)
x2 = np.linspace(-np.pi, np.pi, N)
x1, x2 = np.meshgrid(x1, x2)

x = np.vstack((x1.flatten(), x2.flatten()))

# -------- NORMALIZACIÓN DE X --------
x_mean = x.mean(axis=1, keepdims=True)
x_std  = x.std(axis=1, keepdims=True) + 1e-8
x_norm = (x - x_mean) / x_std

# -------- TARGET --------
y_true = np.sin(x[0]) + 0.5*np.cos(x[1])
y_true = y_true.reshape(1, -1)

# -------- NORMALIZACIÓN DE Y --------
y_mean = y_true.mean()
y_std  = y_true.std() + 1e-8
y_norm = (y_true - y_mean) / y_std
print(y_norm.shape)
# -------- ENTRENAMIENTO --------
mlp = nets.MLP(2, 32, 1)
mlp.train(x_norm, y_norm, lr=1e-3, epochs=6000, batch_size=256)

# -------- PREDICCIÓN --------
y_pred_norm = mlp.forward(x_norm)
y_pred = y_pred_norm * y_std + y_mean

# -------- PLOT --------
extent = [-np.pi, np.pi, -np.pi, np.pi]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Target")
plt.imshow(y_true.reshape(x1.shape), origin="lower", extent=extent)
plt.colorbar()

plt.subplot(1,2,2)
plt.title("MLP")
plt.imshow(y_pred.reshape(x1.shape), origin="lower", extent=extent)
plt.colorbar()

plt.show()
