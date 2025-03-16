from Net_Manager import *
from makeNet import *
np.set_printoptions(precision=3, suppress=True, edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "{0:5.3f}".format(x)))



# TRAIN_SIZE_MNIST = 27455
# TEST_SIZE_MNIST = 7172
# CAM_DATA_SIZE = 26000
# TRAIN_WHOLE = 27455 + 26000


# layers의 묶음 => Network (class)
# Network 을 이용한 학습과 테스트 => NetManager class

###############################################################################################################
#################################                 여기서부터                 #####################################
###############################################################################################################




(x_train, t_train) = load_cam_dataset(file_name='datasets/jung_56_center_200.pkl') # 26000 train
(x_test, t_test) = load_cam_dataset(file_name='datasets/webcam_28.pkl')
batch_mask = np.random.choice(26000, 2048)
x_t, t_t = x_test[batch_mask], t_test[batch_mask]
show_data_img(x_t, t_t, 32)

# netM = NetManager(logisticNet())
# netM.train(x_train, t_train, epoch=100, batch_size=256)
# netM.network.saveNet('weights56/logistic_w0.pkl')
# netM.save_train_log('train_log/logistic_log0.p')
#
# netM = NetManager(NeuralNetL2())
# netM.train(x_train, t_train, epoch=80, batch_size=256)
# netM.network.saveNet('weights56/NeuralNet_w0.pkl')
# netM.save_train_log('train_log/NeuralNet_log0.p')
#
# netM = NetManager(convL2Net())
# netM.train(x_train, t_train, epoch=20, batch_size=128)
# netM.network.saveNet('weights56/CL2_w0.pkl')
# netM.save_train_log('train_log/CL2_log0.p')
#
# netM = NetManager(convL3Net())
# netM.train(x_train, t_train, epoch=20, batch_size=128)
# netM.network.saveNet('weights56/CL3_w0.pkl')
# netM.save_train_log('train_log/CL3_log0.p')

# netM = NetManager(convL4Net())
# netM.train(x_train, t_train, epoch=20, batch_size=64)
# netM.network.saveNet('weights56/CL4_w0.pkl')
# netM.save_train_log('train_log/CL4_log0.p')

# netM = NetManager(convL5Net())
# netM.train(x_train, t_train, epoch=20, batch_size=64)
# netM.network.saveNet('weights56/CL5_w0.pkl')
# netM.save_train_log('train_log/CL5_log0.p')


# netM = NetManager(convL5Net())
# netM.network.loadNet('weights56/CL5_w0.pkl')
# netM.test_ans(x_train, t_train)
# for i in range(4):
# netManager.network.showConvLayer(i)





