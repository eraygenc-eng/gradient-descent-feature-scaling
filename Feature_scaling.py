import numpy as np
import matplotlib.pyplot as plt
import time

x = np.array([[1,40], 
              [2,75], 
              [3,120], 
              [4,250], 
              [5,400]], dtype = float) # Rooms and square_meters

y = np.array([2500000, 3200000, 4500000, 6000000, 7350000], dtype = float) # Price

# =========================
# Without Scaling
# =========================

w1 = np.zeros(2)
b1 = 0.0
lr1 = 0.000001 
epoch = 2000

start1 = time.time()

n1 = len(x)
loss_history1 = []

for i in range(epoch + 1):
    y_pred1 = x @ w1 + b1
    error1 = y_pred1 - y  
    loss1 = (error1 ** 2).mean()

    loss_history1.append(loss1)

    dw1 = (2/n1) * (x.T @ error1)
    db1 = (2/n1) * np.sum(error1)

    w1 = w1 - lr1 * dw1
    b1 = b1 - lr1 * db1

    if i % 100 == 0:
        print(i,loss1,w1,b1)
end1 = time.time()

print("w:", w1)
print("b:", b1)
print("Time: ",end1 - start1)

# =========================
# With Scaling
# =========================

w2 = np.zeros(2)
b2 = 0.0
lr2 = 0.01

start2 = time.time()

n2 = len(x)
loss_history2 = []

mean = np.mean(x,axis = 0)
std = np.std(x, axis = 0)
x_scaled = (x - mean) / std

for i in range(epoch + 1):
    y_pred2 = x_scaled @ w2 + b2
    error2 = y_pred2 - y
    loss2 = (error2 ** 2).mean()

    loss_history2.append(loss2)

    dw2 = (2/n2) * (x_scaled.T @ error2)
    db2 = (2/n2) * np.sum(error2)

    w2 = w2 - lr2 * dw2
    b2 = b2 - lr2 * db2

    if i % 10 == 0:
        print(i,loss2,w2,b2)

end2 = time.time()
print("w:", w2)
print("b:", b2)
print("Time: ",end2 - start2)

# =========================
# Graphic
# =========================


plt.plot(loss_history1, label="No Scaling")
plt.plot(loss_history2, label="With Scaling")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.yscale("log")   

plt.savefig("feature_scaling_comparison.png", dpi=300, bbox_inches="tight")

plt.show()