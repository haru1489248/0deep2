"""
RNNの時間方向の逆伝播では、同じ重み行列 Wh.T を何度も掛ける。

Wh の最大特異値が 1 より大きいと、
勾配ノルムが増加しやすく、勾配爆発が起こりやすい。

Wh の最大特異値が 1 より小さいと、
勾配ノルムが減少しやすく、勾配消失が起こりやすい。

最大特異値は、行列がベクトルの長さを最大で何倍に伸ばすかを表す。
これは A の L2ノルム（スペクトルノルム）と一致し、
||A||_2 = sqrt(λ_max(A^T A)) = σ_max(A)
で求まる。
"""

import numpy as np
import matplotlib.pyplot as plt

N = 2 # バッチサイズ
H = 3 # 隠れ状態ベクトルの次元数
T = 20 # 時系列データの長さ

dh = np.ones((N, H))
np.random.seed(3) # 再現性のため乱数のシードを固定
Wh = np.random.randn(H, H)

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# グラフの描画
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()
