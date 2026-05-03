import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0]) # 共起行列
# [0 1 0 0 0 0 0]
print(W[0]) # PPMI行列
# [0.        1.8073549 0.        0.        0.        0.        0.       ]
print(U[0]) # SVD
# [ 3.4094876e-01  0.0000000e+00  1.2051624e-01 -3.6082248e-16  -1.1102230e-16  9.3232495e-01  1.6260980e-16]

# 次元削減して二次元にする
print(U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
