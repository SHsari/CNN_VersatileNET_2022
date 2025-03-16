import cv2
from utils import *
from Network import Network


class NetManager:
    # 트레이너 클래스 입니다.
    # 트레인 데이터, 훈련시킬 네트워크, 에폭, 배치사이즈를 인수로 받고
    # 네트워크의 훈련과 테스트 기능을 제공합니다.
    def __init__(self, network):

        self.network = network
        self.epoch = None
        self.batch_size = None
        self.cost = None
        self.accu = None

    def train(self, x_train, t_train, epoch, batch_size):
        train_size = t_train.shape[0]
        self.epoch = epoch
        self.batch_size = batch_size
        iter_per_epoch = train_size // batch_size
        self.cost = np.full(epoch * iter_per_epoch, -1, float)
        self.accu = np.full(epoch * iter_per_epoch, -1, float)

        print('Training Initiate', end=' ')
        it = -1
        for i in range(self.epoch):
            for j in range(iter_per_epoch):
                it += 1
                batch_mask = np.random.choice(train_size, self.batch_size)
                x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
                pred = self.network.get_prediction(x_batch, train_flag=True)
                self.cost[it] = self.network.get_loss(pred, t_batch)
                self.accu[it] = get_accuracy(pred, t_batch)
                self.network.back_propagation()
                if j % 4 == 3:
                    print('.', end='')
            print('')
            print("epoch {0:3d},  batch_accu: {1:10.8f}".format(i + 1, self.accu[it]), end=' ')
        print('')

    def test_ans(self, x, t):
        print('test initiate')
        pred = self.network.get_prediction(x, train_flag=False)
        accuracy = get_accuracy(pred, t)
        print('accuracy: {}'.format(accuracy))
        return np.argmax(pred, axis=1), accuracy

    # 카메라와 연동 할 경우 즉각적인 결과값 반환을 위해
    # test 함수와 유사한 get abc 함수를 추가로 작성했습니다.

    def save_train_log(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.cost, file)
            pickle.dump(self.accu, file)
            pickle.dump(self.epoch, file)

    def load_train_log(self, file_name):
        with open(file_name, 'rb') as file:
            self.cost = pickle.load(file)
            self.accu = pickle.load(file)
            self.epoch = pickle.load(file)

    def makePlot(self):
        x = range(len(self.accu))
        plt.figure()
        plt.plot(x, self.cost, label='cost')
        plt.plot(x, self.accu, label='accu')
        plt.xlabel('iteration')
        plt.title('epoch = {}'.format(self.epoch))
        plt.legend()
        plt.show


    def get_abc(self, x):
        # 카메라에서 들어온 데이터를 기반으로 결과값을 판단하는 함수입니다.
        # 카메라 데이터가 n개씩 묶여 들어온다고 가정합니다.
        pred = self.network.get_chance(x)
        pred = np.mean(pred, axis=0)  # 묶여 들어온 데이터 값의 평균을 냅니다. 하나의 결과값으로 추론하기 위해.
        max_chance = np.max(pred)
        # 확률이 50% 이상이면 26개 중 하나의 알파뱃 번호를 리턴합니다. (0~25)
        # 하나의 알파뱃으로 특정되지 않을 경우 확률은 0, 번호는 -1을 리턴합니다.
        if max_chance > 0.7:
            return max_chance, np.argmax(pred)
        else:
            return 0.0, -1

    def camera_out(self):
        ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        cam = cv2.VideoCapture(0)
        it = 0
        test_data = []
        while True:
            it += 1
            ret, frame = cam.read()

            x1 = 760
            x2 = 1160
            y1 = 340
            y2 = 740

            # define region of interest
            roi = frame[y1:y2, x1:x2]
            cv2.imshow('roi', roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (56, 56), interpolation=cv2.INTER_AREA)

            copy = frame.copy()
            cv2.rectangle(copy, (x1, y1), (x2, y2), (255, 0, 0), 5)

            roi = roi.reshape(1, 1, 56, 56)
            roi = roi.astype(np.float32)
            roi /= 255.0

            chance, ABCarg = self.get_abc(roi)


            copy = cv2.flip(copy, 1)
            if ABCarg >= 0:
                str = ABC[ABCarg] + " {0:5.3f}%".format(chance * 100)
                cv2.putText(copy, str, (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            cv2.imshow('frame', copy)

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break

        cam.release()
        cv2.destroyAllWindows()
