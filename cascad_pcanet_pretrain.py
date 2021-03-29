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

from multi_modal_pcanet.cascade_pcanet.pcanet_func.utils import save_model

def train(images_train, y_train):
    images_train_rgb, images_train_dep, images_train_ir = images_train
    print("[INFO]Training PCANet")
    pcanet_ir = net.PCANet1(
        image_shape=28,
        filter_shape_l1=3, step_shape_l1=1, n_l1_output=3,
        filter_shape_pooling=4, step_shape_pooling=2
    )
    pcanet_dep = net.PCANet1(
        image_shape=28,
        filter_shape_l1=3, step_shape_l1=1, n_l1_output=3,
        filter_shape_pooling=4, step_shape_pooling=2
    )
    pcanet_rgb = net.PCANet1(
        image_shape=28,
        filter_shape_l1=3, step_shape_l1=1, n_l1_output=3,
        filter_shape_pooling=4, step_shape_pooling=2
    )
    pcanet_dep.fit(images_train_dep)
    pcanet_rgb.fit(images_train_rgb)
    pcanet_ir.fit(images_train_ir)
    X_train_rgb = pcanet_rgb.transform(images_train_rgb)
    X_train_ir = pcanet_ir.transform(images_train_ir)
    X_train_dep = pcanet_dep.transform(images_train_dep)
    print(X_train_dep.shape)
    print(X_train_ir.shape)
    print(X_train_rgb.shape)
    print("training classifier:")
    print("[INFO]Training the classifier_dep ...")
    classifier_dep = SVC(C=20, probability=True)
    classifier_dep.fit(X_train_dep, y_train)
    print("[INFO]Training the classifier_ir ...")
    classifier_ir = SVC(C=20, probability=True)
    classifier_ir.fit(X_train_ir, y_train)
    classifier_rgb = SVC(C=20, probability=True)
    classifier_rgb.fit(X_train_rgb, y_train)
    return pcanet_rgb, classifier_rgb, pcanet_ir, classifier_ir, pcanet_dep, classifier_dep

def load_data():
    images_train_rgb = np.load('../casia_surf_data/data_train_rgb.npy')
    images_train_dep = np.load('../casia_surf_data/data_train_dep.npy')
    images_train_ir = np.load('../casia_surf_data/data_train_ir.npy')

    print(images_train_rgb.shape)
    y_train = np.load('data_train_label.npy')

    images_train_rgb = images_train_rgb / 255
    images_train_dep = images_train_dep / 255
    images_train_ir = images_train_ir / 255
    images_test_rgb = np.load('../casia_surf_data/data_val_rgb.npy')
    images_test_dep = np.load('../casia_surf_data/data_val_dep.npy')
    images_test_ir = np.load('../casia_surf_data/data_val_ir.npy')
    y_test = np.load('../casia_surf_data/data_val_label.npy')

    # images_test_dep=images_test_dep.astype(np.float32)
    # images_test_ir=images_test_ir.astype(np.float32)
    images_test_rgb = images_test_rgb / 255
    images_test_dep = images_test_dep / 255
    images_test_ir = images_test_ir / 255
    # /255 is better than astype
    images_train = images_train_rgb, images_train_dep, images_train_ir
    images_test = images_test_rgb, images_test_dep, images_test_ir
    return images_train, y_train, images_test, y_test




print("[INFO]Loading the model...")
images_train, y_train, images_test, y_test = load_data()

print("[INFO]Training the model...")
pcanet_rgb, classifier_rgb, pcanet_ir, classifier_ir, pcanet_dep, classifier_dep = train(images_train, y_train)
print("[INFO]Testing the model...")


save_model(classifier_rgb, join("pretrained_model", "classifier_rgb_cascade.pkl"))
save_model(classifier_ir, join("pretrained_model", "classifier_ir_cascade.pkl"))
save_model(classifier_dep, join("pretrained_model", "classifier_dep_cascade.pkl"))
save_model(pcanet_rgb, join("pretrained_model","pcanet_rgb_cascade.pkl"))
save_model(pcanet_ir, join("pretrained_model", "pcanet_ir_cascade.pkl"))
save_model(pcanet_dep, join("pretrained_model", "pcanet_dep_cascade.pkl"))
