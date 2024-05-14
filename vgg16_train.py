import numpy as np
from sklearn.datasets import fetch_lfw_people
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
import os
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

from datetime import datetime

def preprocessing_predict_img(gray_img):
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    resized_img = cv2.resize(color_img, (224, 224))
    resized_img = np.expand_dims(resized_img, axis=0)

    return resized_img
def preprocessing_img(faces, ids):
    x_faces = []
    for i in range(len(faces)):
        gray_img = np.array(faces[i], dtype=np.uint8)

        color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        resized_img = cv2.resize(color_img, (224,  224))
        x_faces.append(resized_img)

    x_faces = np.array(x_faces)
    print(x_faces.shape)

    y_faces = to_categorical(ids - 1)

    return x_faces, y_faces

def build_model(face_images, class_count, model_name):
    # vgg 16 model
    classifier_vgg16 = VGG16(input_shape=face_images.shape[1:], include_top=False, weights='imagenet')

    # not train top layers
    for layer in classifier_vgg16.layers:
        layer.trainable = False

    model = classifier_vgg16.output
    model = GlobalAveragePooling2D()(model)
    model = Dense(1024,activation='relu')(model)
    model = Dense(1024,activation='relu')(model)
    model = Dense(512,activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(class_count,activation='softmax')(model)

    model = Model(inputs=classifier_vgg16.input, outputs=model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg_main(faces, ids):
    class_counts = len(np.unique(ids))

    x_faces, y_faces = preprocessing_img(faces, ids)

    x_train, x_val, y_train, y_val = train_test_split(x_faces, y_faces,
                                                      test_size=0.2,
                                                      stratify=y_faces,
                                                      random_state=0)
    model_name = 'Face_trained_model_' + datetime.now().strftime("%H_%M_%S_")
    model = build_model(x_faces, class_counts, model_name)
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, batch_size=5)
    print('Train Done!!')
    return model

def predict_classes(model, gray_img, batch_size=5, verbose=1):
    proba = model.predict(gray_img, batch_size=batch_size, verbose=verbose)
    id = proba.argmax(axis=-1)[0] + 1
    pred = proba[0].max()

    return id, pred

