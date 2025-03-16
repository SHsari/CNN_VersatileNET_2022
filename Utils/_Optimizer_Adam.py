import numpy as np

class Adam:
    def __init__(self, W, b, lr=0.001, beta_v=0.9, beta_s=0.999):
        self.lr = lr
        self.beta_v = beta_v
        self.beta_s = beta_s
        self.iter = 0
        self.vW, self.sW = np.zeros_like(W), np.zeros_like(W)
        self.vb, self.sb = np.zeros_like(b), np.zeros_like(b)

    def update(self, gW, gb):
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta_s ** self.iter) / (1.0 - self.beta_v ** self.iter)

        self.vW += (1 - self.beta_v) * (gW - self.vW)
        self.sW += (1 - self.beta_s) * (gW ** 2 - self.sW)
        dW = lr_t * self.vW / (np.sqrt(self.sW) + 1e-7)

        self.vb += (1 - self.beta_v) * (gb - self.vb)
        self.sb += (1 - self.beta_s) * (gb ** 2 - self.sb)
        db = lr_t * self.vb / (np.sqrt(self.sb) + 1e-7)

        return dW, db


    # def SGD(self, lr, dout):
    #     self.W -= lr * np.dot(self.x.T, dout)
    #     self.b -= lr * np.sum(dout, axis=0)

    # def AdaGrad(self, lr, dout):
    #     dW = np.dot(self.x.T, dout)
    #     self.hW += dW**2
    #     self.W -= lr * dW / (np.sqrt(self.hW) + 1e-7)
    #     db = np.sum(dout, axis=0)
    #     self.hb += db**2
    #     self.b -= lr * db / (np.sqrt(self.hb) + 1e-7)
    #
    # def Adam(self, lr, dout):
    #     if self.m is None:
    #         self.m, self.v = {}, {}
    #         for key, v

