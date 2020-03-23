import numpy as np

v1 = np.random.random(size=(3,))
v2 = np.random.random(size=(3,))

# print(v1,v2)
# print(v1*v2) # 向量的对应项相乘
# print(np.sum(v1 * v2))

v3 = np.array([2,2,2])
print(np.linalg.norm(v3))
