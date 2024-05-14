import numpy as np
from sklearn.datasets import fetch_lfw_people
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from datetime import datetime
from tensorflow.keras.models import load_model

def load_faces():
    faces = fetch_lfw_people(min_faces_per_person=100,
                             resize=1.0,
                             slice_=(slice(60, 188),
                                     slice(60, 188)),
                             color=True)
    return faces

def balance_data(faces):
    mask = np.zeros(faces.target.shape, dtype=np.bool_)

    for target in np.unique(faces.target):
        mask[np.where(faces.target == target)[0][:100]] = 1

    x_faces = faces.data[mask]
    y_faces = faces.target[mask]
    x_faces = np.reshape(x_faces,
                         (x_faces.shape[0], faces.images.shape[1], faces.images.shape[2], faces.images.shape[3]))
    print(x_faces.shape)
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

if __name__ == '__main__':
    faces = load_faces()
    class_counts = len(faces.target_names)
    # print(faces.target_names)
    # print(faces.images.shape)

    x_faces, y_faces = balance_data(faces)

    face_images = x_faces
    face_labels = to_categorical(y_faces)

    x_train, x_val, y_train, y_val = train_test_split(face_images, face_labels,
                                                      test_size=0.15,
                                                      stratify=face_labels,
                                                      random_state=0)
    model_name = 'Face_trained_model_' + datetime.now().strftime("%H_%M_%S_")
    model = build_model(face_images, class_counts, model_name)
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=25)
    print('Done!!')





