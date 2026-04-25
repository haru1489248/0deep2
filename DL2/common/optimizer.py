class SGD:
    """
    lr: learning rate（学習率）
    """
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
