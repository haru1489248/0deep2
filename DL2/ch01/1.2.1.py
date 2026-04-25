import numpy as np

W1 = np.random.randn(2, 4) # 重み
b1 = np.random.randn(4) # バイアス
x = np.random.randn(10, 2) # 入力
h = np.dot(x, W1) + b1 # ニューロンを計算する

# sigmoid関数の実装(1.5)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 活性化関数によって非線形にすることでアクティベーションを作る
a = sigmoid(h)

# 隠れ層のニューロンが４つ、出力層のニューロンは三つなので4×3の行列を作成する
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

s = np.dot(a, W2) + b2
print(s)
print(s.shape)
