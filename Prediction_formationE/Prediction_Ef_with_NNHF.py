
#### Written by Jeongrae Kim in KIST

import tensorflow as tf
import numpy as np
from NNHF import CNNs
from math import sqrt

filenum = 1

tf.set_random_seed(123)
np.random.seed(123)

model_log_ = './Pretrained_model/'
file_name_input_test_x = './Data/'+ str(filenum) +'_test_Ef.dat'
file_name_test_MAE = './Results/'+ str(filenum) +'_test_MAE.out'
file_name_test_Y_real = './Results/'+ str(filenum) +'_test_Y_real.out'
file_name_test_Y_hat = './Results/'+ str(filenum) +'_test_Y_hat.out'

test_x_list = []
with open(file_name_input_test_x) as f:
    for line in f:
        tmp = line.strip().split(",")
        test_x_list.append([tmp[0], tmp[1], tmp[2:]])

test_num = len(test_x_list)
test_acc = 0
test_Y_real_E, test_Y_hat_E = [], []
test_ID = []

with tf.Session() as sess:
    model = CNNs(sess)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint_E = tf.train.latest_checkpoint(model_log_)
    if checkpoint_E:
        saver.restore(sess, checkpoint_E)
        print("Restore weights of networks ~")

    test_mae_sum = 0
    test_rmse_sum = 0
    for i in range(len(test_x_list)):
        test_x_data, p_Ef, AB_material_vactor = CNNs.data_label_test(test_x_list[i:(i+1)], 1)
        test_p_Ef = np.reshape(p_Ef, [-1, 1])
        test_AB_material_vector = np.reshape(AB_material_vactor, [-1, 60,1,1])
        tst_MAE, tst_MSE, tst_y_hat, tst_y_real = model.test_step(test_x_data, test_AB_material_vector, test_p_Ef)
        test_mae_sum += tst_MAE
        test_rmse_sum += tst_MSE
        test_Y_real_E.append(tst_y_real[0])
        test_Y_hat_E.append(tst_y_hat[0])

    print("Test MAE :  ", test_mae_sum / test_num)
    print("Test RMSE :  ", sqrt(test_rmse_sum / test_num))

with open(file_name_test_MAE, 'w') as f:
    f.write(str(test_mae_sum / test_num) + "\n")
Y_file_ = open(file_name_test_Y_real, 'w')
Y_file_.writelines(["%f\n" % float(item) for item in test_Y_real_E])
Y_file_.close()
Y_file = open(file_name_test_Y_hat, 'w')
Y_file.writelines(["%f\n" % float(item) for item in test_Y_hat_E])
Y_file.close()

