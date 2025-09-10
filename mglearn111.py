import mglearn as mg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Forge
# x,y = mg.datasets.make_forge()
# print("Forge dataset shape: ", x.shape)
# print(mg.discrete_scatter(x[:,0],x[:,1],y))
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# plt.title("Forge Dataset Visualization")
# mg.discrete_scatter(x[:,0], x[:,1], y)
# plt.show()

# # Wave
# x,y = mg.datasets.make_wave()
# print("Forge dataset shape: ", x.shape)
# print("Forge labels: ", y)
# plt.plot(x,y,'v')
# plt.plot(x,y,'o', label='Circle')
# plt.plot(x,y+0.7,'^', label='Square')
# plt.plot(x,y+0.5,'^', label='Triangle')
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.title("Wave Dataset")
# plt.legend()
# plt.show()

# # Wave Dataset Line
x,y = mg.datasets.make_wave()
plt.plot(x,y, 'o', label='Data points')
coeffs = np.polyfit(x.flatten(),y,1)
y_fit = np.polyval(coeffs,x)
plt.plot(x,y_fit, color='red', label='Linear fit')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Wave Dataset Line")
plt.legend()
plt.show()