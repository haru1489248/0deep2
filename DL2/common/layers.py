import numpy as np
from common.functions import softmax, cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW # gradsの参照位置を変更せず、値をdWのものと同じにする
        return dx # 前の層にdoutとして誤差を伝播するために戻り値として返している

"""
各レイヤーの grads 配列の参照を、ネットワーク側の grads に一度だけ集める。

その後は grads[0][...] = dW によって同じ配列の中身だけ更新する。

参照先は変わらないので、ネットワーク側の grads も常に最新の値を見られる。
"""

# NOTE: sigmoid 関数の微分は y(1 - y)
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None # sigmoidの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None # softmaxの出力（確率）
        self.t = None # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size # 勾配を平均化するためにバッチサイズで割る

        return dx

# one-hotベクトルとの行列積を計算する代わりにインデックスを指定して行ベクトルを取り出す層
class Embedding:
    """
    idx: 抽出する行のインデックス（単語ID）をnumpy配列として格納する
    """
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 # 要素を全て0にする

        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]

        # もしくは
        # np.add.at(dW, self.idx, dout)
        # np.add.at(A, idx, B)はidxで指定した位置に対して、AにBを加算する

        return None
