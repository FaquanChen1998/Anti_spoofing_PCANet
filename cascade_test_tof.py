from os.path import join

import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
import multi_modal_pcanet.cascade_pcanet.pcanet_1 as net
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from multi_modal_pcanet.cascade_pcanet.pcanet_func.utils import save_model, load_model

def load_data():

    images_train_dep = np.load('../dm_tof_data/data_train_dep_tof.npy')
    images_train_ir = np.load('../dm_tof_data/data_train_ir_tof.npy')


    y_train = np.load('../dm_tof_data/data_label_train_tof.npy')


    images_train_dep = images_train_dep / 255
    images_train_ir = images_train_ir / 255
    # images_test_rgb = np.load('data_test_rgb.npy')
    images_test_dep = np.load('../dm_tof_data/data_test_dep_tof.npy')
    images_test_ir = np.load('../dm_tof_data/data_test_ir_tof.npy')
    y_test = np.load('../dm_tof_data/data_label_test_tof.npy')

    # images_test_dep=images_test_dep.astype(np.float32)
    # images_test_ir=images_test_ir.astype(np.float32)
    # images_test_rgb = images_test_rgb / 255
    images_test_dep = images_test_dep / 255
    images_test_ir = images_test_ir / 255
    # /255 is better than astype
    images_train = images_train_dep, images_train_ir
    images_test = images_test_dep, images_test_ir
    return images_train, y_train, images_test, y_test




print("[INFO]Loading the model...")
images_train, y_train, images_test, y_test = load_data()

pcanet_ir = load_model(join("pretrained_model", "pcanet_ir_cascade_tof.pkl"))
pcanet_dep = load_model(join("pretrained_model", "pcanet_dep_cascade_tof.pkl"))
classifier_ir = load_model(join("pretrained_model", "classifier_ir_cascade_tof.pkl"))
classifier_dep = load_model(join("pretrained_model", "classifier_dep_cascade_tof.pkl"))
# pcanet_rgb = load_model(join("result", "pcanet_rgb_cascade.pkl"))
# classifier_rgb = load_model(join("result", "classifier_rgb_cascade.pkl"))
print(images_test[1].shape)
# dep = pcanet_dep.transform(images_test[1])
# y_pred = classifier_dep.predict(dep)
pred = []
for i in range(images_test[1].shape[0]):
    test_dep = images_test[0][i]
    test_ir = images_test[1][i]
    test_dep = test_dep[np.newaxis, :, :, np.newaxis]
    test_ir = test_ir[np.newaxis, :, :, np.newaxis]
    X_test_dep = pcanet_dep.transform(test_dep)
    y_pred_dep_p = classifier_dep.predict_proba(X_test_dep)
    # print(y_test[i].astype)
    # print(y_test[i])
    if int(y_test[i]) == 0:
        actual = "Fake"
    else:
        actual = "Real"

    if float(y_pred_dep_p[0][0]) >= 0.65:
        y_pred = 0
        pred.append(y_pred)
        print("Pred: Fake, Actual:" + actual)
    else:
        if float(y_pred_dep_p[0][1]) >= 0.90:
            print("Pred: Real, Actual:" + actual)
            y_pred = 1
            pred.append(y_pred)
        else:
            X_test_ir = pcanet_ir.transform(test_ir)
            X_test_rgb = pcanet_ir.transform(test_ir)
            y_pred_ir_p = classifier_ir.predict_proba(X_test_ir)
            # y_pred_rgb_p = classifier_rgb.predict_proba(X_test_rgb)
            y_pred =(y_pred_dep_p[0][1]+y_pred_ir_p[0][1])/2
            if y_pred >= 0.5:
                print("Pred: Real, Actual:" + actual)
                y_pred = 1
                pred.append(y_pred)
            else:
                print("Pred: Fake, Actual:" + actual)
                y_pred = 0
                pred.append(y_pred)


        # y_pred = int(y_pred_ir)

y_pred = np.array(pred)
y_test = np.array(y_test)
y_test =y_test.astype(int)
print(y_pred.shape)
print(y_test.shape)

print("[Result]accuracy:")
target_name = ['fake', 'real']
print(classification_report(y_test, y_pred, digits=4, target_names=target_name))
