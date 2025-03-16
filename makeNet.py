from Utils._Layers import *
from utils import *
from Network import Network


def convL5Net():
    # 5개의 Convolution Layer를 가진 네트워크를 반환합니다.
    # 각 레이어의 하이퍼 파라미터는 이 함수 안에서 수정해주시면 됩니다.
    # 아래에 더 단순한 네트워크를 만들어주는 함수가 있습니다.

    conv1 = make_conv_layer(fnum=16, input_ch=1, fsize=5, stride=1, pad=2)  # out: 56*56*16
    pool1 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 28*28*16
    conv2 = make_conv_layer(fnum=32, input_ch=16, fsize=5, stride=1, pad=2) # out: 28*28*32
    pool2 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 14*14*32
    conv3 = make_conv_layer(fnum=32, input_ch=32, fsize=3, stride=1, pad=1) # out: 14*14*32
    conv4 = make_conv_layer(fnum=32, input_ch=32, fsize=3, stride=1, pad=1) # out: 14*14*32
    conv5 = make_conv_layer(fnum=16, input_ch=32, fsize=3, stride=1, pad=1) # out: 14*14*16
    pool3 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 7*7*16

    aff1 = make_affine_layer(input_size=7*7*16, out_size=128)
    aff2 = make_affine_layer(input_size=128, out_size=26)

    layers = []
    layers.append(conv1);   layers.append(Relu());  layers.append(pool1)
    layers.append(conv2);   layers.append(Relu());  layers.append(pool2)
    layers.append(conv3);   layers.append(Relu())
    layers.append(conv4);   layers.append(Relu())
    layers.append(conv5);   layers.append(Relu());  layers.append(pool3)
    layers.append(Dropout(dropout_ratio=0.4))
    layers.append(aff1);    layers.append(Relu())
    layers.append(Dropout(dropout_ratio=0.4))
    layers.append(aff2)

    # Network 클래스는 레이어들이 순서대로 들어간 리스트를 인자로 받아
    # 해당 레이어들로 이루어진 네트워크를 만들어줍니다.
    return Network(layers)


def convL4Net():
    # 4개의 Convolution Layer를 가진 네트워크를 반환합니다.
    # 각 레이어의 하이퍼 파라미터는 이 함수 안에서 수정해주시면 됩니다.
    # 아래에 더 단순한 네트워크를 만들어주는 함수가 있습니다.

    conv1 = make_conv_layer(fnum=16, input_ch=1, fsize=5, stride=1, pad=2)  # out: 56*56*16
    pool1 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 28*28*16
    conv2 = make_conv_layer(fnum=32, input_ch=16, fsize=5, stride=1, pad=2) # out: 28*28*32
    pool2 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 14*14*32
    conv3 = make_conv_layer(fnum=32, input_ch=32, fsize=3, stride=1, pad=1) # out: 14*14*32
    conv4 = make_conv_layer(fnum=16, input_ch=32, fsize=3, stride=1, pad=1) # out: 14*14*16
    pool3 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 7*7*16

    aff1 = make_affine_layer(input_size=7*7*16, out_size=128)
    aff2 = make_affine_layer(input_size=128, out_size=26)

    layers = []
    layers.append(conv1);   layers.append(Relu());  layers.append(pool1)
    layers.append(conv2);   layers.append(Relu());  layers.append(pool2)
    layers.append(conv3);   layers.append(Relu())
    layers.append(conv4);   layers.append(Relu());  layers.append(pool3)
    layers.append(Dropout(dropout_ratio=0.25))
    layers.append(aff1);    layers.append(Relu());
    layers.append(Dropout(dropout_ratio=0.25))
    layers.append(aff2)

    # Network 클래스는 레이어들이 순서대로 들어간 리스트를 인자로 받아
    # 해당 레이어들로 이루어진 네트워크를 만들어줍니다.
    return Network(layers)


def convL3Net():
    # 2개의 Convolution layer을 사용하는 network를 리턴합니다.
    conv1 = make_conv_layer(fnum=16, input_ch=1, fsize=5, stride=1, pad=2)  # out: 56*56*16
    pool1 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 28*28*16
    conv2 = make_conv_layer(fnum=32, input_ch=16, fsize=5, stride=1, pad=2) # out: 28*28*32
    pool2 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 14*14*32
    conv3 = make_conv_layer(fnum=16, input_ch=32, fsize=3, stride=1, pad=1) # out: 14*14*16
    pool3 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 7*7*16

    aff1 = make_affine_layer(input_size=7*7*16, out_size=128)
    aff2 = make_affine_layer(input_size=128, out_size=26)

    layers = []
    layers.append(conv1);   layers.append(Relu());  layers.append(pool1)
    layers.append(conv2);   layers.append(Relu());  layers.append(pool2)
    layers.append(conv3);   layers.append(Relu());  layers.append(pool3)
    layers.append(Dropout(dropout_ratio=0.3))
    layers.append(aff1);    layers.append(Relu())
    layers.append(Dropout(dropout_ratio=0.3))
    layers.append(aff2)

    return Network(layers)


def convL2Net():
    # 2개의 Convolution layer을 사용하는 network를 리턴합니다.
    #56x56
    conv1 = make_conv_layer(fnum=16, input_ch=1, fsize=5, stride=1, pad=0)  # out: 52*52*16
    pool1 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 26*26*16
    conv2 = make_conv_layer(fnum=32, input_ch=16, fsize=5, stride=1, pad=1) # out: 24*24*32
    pool2 = Pooling(pool_h=2, pool_w=2, stride=2)                           # out: 12*12*32

    aff1 = make_affine_layer(input_size=12*12*32, out_size=128)
    aff2 = make_affine_layer(input_size=128, out_size=26)

    layers = []
    layers.append(conv1);   layers.append(Relu());  layers.append(pool1)
    layers.append(conv2);   layers.append(Relu());  layers.append(pool2)
    layers.append(Dropout(dropout_ratio=0.3))
    layers.append(aff1);    layers.append(Relu())
    layers.append(Dropout(dropout_ratio=0.3))
    layers.append(aff2)

    return Network(layers)

def NeuralNetL2():
    aff1 = make_affine_layer(input_size=3136, out_size=128)
    aff2 = make_affine_layer(input_size=128, out_size=26)

    layers = []
    layers.append(aff1);    layers.append(Relu());
    layers.append(Dropout(dropout_ratio=0.2))
    layers.append(aff2);

    return Network(layers)

def logisticNet():
    aff1 = make_affine_layer(input_size=3136, out_size=26)
    layers = []
    layers.append(Dropout(dropout_ratio=0.2))
    layers.append(aff1);

    return Network(layers)
