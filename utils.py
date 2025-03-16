from Utils._Layers import Affine, Convolution
import matplotlib.pyplot as plt
import numpy as np
import pickle


def make_affine_layer(input_size, out_size):
    W = np.random.randn(input_size, out_size) * np.sqrt(2 / input_size)
    b = np.zeros(out_size)
    return Affine(W, b)


def make_conv_layer(fnum, input_ch, fsize, stride, pad):
    W = np.random.randn(fnum, input_ch, fsize, fsize) * np.sqrt(2 / (input_ch * (fsize ** 2)))
    # W = np.random.randn(fnum, input_ch, fsize, fsize) * 0.01
    b = np.zeros(fnum)
    return Convolution(W, b, stride, pad)


def get_accuracy(pred, t):
    y = np.argmax(pred, axis=1)
    t = np.argmax(t, axis=1)
    return np.sum(y == t) / y.shape[0]


# dataset을 이미지로 보여주는 함수 입니다.
def img_show(batch):
    max_view = 32
    current_view = 1

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)
    if batch.ndim == 4:
        A, B, C, D = batch.shape
        batch = batch.reshape(A,C,D)
    for img in batch:
        ax = fig.add_subplot(4, 8, current_view, xticks=[], yticks=[])
        ax.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        current_view += 1
        if current_view > max_view:
            break

    plt.show()


def show_data_img(x, t, batch_number):
    batch_mask = np.random.choice(x.shape[0], batch_number)
    print(np.argmax(t[batch_mask], axis=0))
    img_show(x[batch_mask])
    print(batch_number)
    return 0


def load_mnist_dataset(file_name="datasets/sign_mnist.pkl"):
    with open(file_name, 'rb') as f:
        x_train = pickle.load(f)
        t_train = pickle.load(f)
        x_test = pickle.load(f)
        t_test = pickle.load(f)
        return (x_train, t_train), (x_test, t_test)


def load_cam_dataset(file_name="datasets/webcam_28.pkl"):
    with open(file_name, 'rb') as f:
        (x_train, t_train) = pickle.load(f)
    return x_train, t_train


def load_dataset():
    (x_tr_mnist, t_tr_mnist), (x_test, t_test) = load_mnist_dataset()
    x_Cam, t_Cam = load_cam_dataset()
    x_train = np.concatenate((x_tr_mnist, x_Cam), axis=0)
    t_train = np.concatenate((t_tr_mnist, t_Cam), axis=0)
    return (x_train, t_train), (x_test, t_test)

def convFilterShow():
    return


