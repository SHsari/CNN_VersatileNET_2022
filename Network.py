from Utils._Layers import *
from utils import img_show
from Utils._functions import softmax


class Network:

    # get_prediction: batch_size * 26의 np.array 반환
    # softmax(get_prediction(x)) 를 하면 각 데이터별로 26클래스 마다의 확률이 나온다.
    # SaveNet LoadNet 추가
    def __init__(self, layers):
        self.layers = layers
        self.SM_CE = SoftmaxWithLoss()

    def get_prediction(self, x, train_flag=False):
        if train_flag:
            for layer in self.layers:
                if isinstance(layer, Dropout):
                    x = layer.forward(x, train_flag)
                else:
                    x = layer.forward(x)
        else:
            for layer in self.layers:
                x = layer.forward(x)
        return x

    def get_loss(self, pred, t):
        return self.SM_CE.forward(pred, t)

    def back_propagation(self):
        dout = 1
        dout = self.SM_CE.backward(dout)
        layers_r = list(self.layers)
        layers_r.reverse()
        for layer in layers_r:
            dout = layer.backward(dout)
        return dout

    def get_chance(self, x):
        for layer in self.layers:
            x = layer.forward(x)  # train_flag 는 default 로 False 입니다.
        return softmax(x)

    def saveNet(self, file_name):
        with open(file_name, 'wb') as file:
            for layer in self.layers:
                if isinstance(layer, Convolution) or \
                        isinstance(layer, Pooling) or isinstance(layer, Affine):
                    layer.save(file)

    def loadNet(self, file_name):
        with open(file_name, 'rb') as file:
            for layer in self.layers:
                if isinstance(layer, Convolution) or \
                        isinstance(layer, Pooling) or isinstance(layer, Affine):
                    layer.load(file)

    def showConvLayer(self, numb):
        tmpConvLayer= []
        for layer in self.layers:
            if isinstance(layer, Convolution):
                tmpConvLayer.append(layer)

        conv = tmpConvLayer[numb]
        if isinstance(conv, Convolution):
            WW = softmax(np.mean(conv.W, axis=1))
            img_show(WW)
        else:
            print("Layer Number '{}' is not Appropriate".format(numb))
