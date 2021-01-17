import numpy as np
from sklearn.preprocessing import normalize

# a = np.array([[1, 2], [3, 4] , [5,6]])
# b = np.array([[3, 3], [3, 4] , [5,6]])
# # print(np.argpartition(b, 3))
# # c = b.repeat(10).reshape((-1,10)).transpose()
# print(a)
# print(b)
# # # print(c)
# # print(a/b.reshape(a.shape[0],-1))
# print(np.divide(a,b))
init_topk = 5
a = np.array([[1, 2, 3], [4, 10, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])

print(a)
a = normalize(a, axis=1, norm='l1')
print(a)


# c = a / b.reshape(a.shape[0], -1)

# print(a)
# print(b)
# print(c)