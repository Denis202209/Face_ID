import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np
import mysql.connector
from vgg16_train import vgg_main, preprocessing_predict_img, predict_classes
from keras.models import load_model

import img_navigator
from datetime import datetime

window = tk.Tk()
window.title("Face recognition system")
# window.config(background="lime")
l1 = tk.Label(window, text="Name", font=("Courier New", 22))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Courier New", 22))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Courier New", 22))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)


def train_classifier():
    data_dir = "./data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = vgg_main(faces, ids)

    # clf = cv2.face.LBPHFaceRecognizer_create()
    # clf.train(faces, ids)
    clf.save("classifier.keras")
    messagebox.showinfo('Result', 'Training dataset completed!!!')


b1 = tk.Button(window, text="Training", font=("Courier New", 20), bg="#B6B6B6", fg="#006600", command=train_classifier)
b1.grid(column=0, row=5)


def detect_face():
    #def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            predict_img = preprocessing_predict_img(gray_image[y:y + h, x:x + w])

            id, pred = predict_classes(clf, predict_img)
            print(f'----- ID is {id}')
            print(f'----- pred is {pred}')
            
            confidence = pred

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="root",
                database="Authorized_user"
            )
            mycursor = mydb.cursor()
            mycursor.execute("select name from my_table where id=" + str(id))
            s = mycursor.fetchone()
            print(f' ----- S is {s}')
            if s:
                s = '' + ''.join(s)
            else:
                s = 'NAME_NOT_FOUND'

            # авторизованный пользователь, зеленая рамка с именем
            if confidence > 0.74:
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                # Красный свет
                # Добавлено: 1) Красная рамка поярче
                # 2) сохранение изображения в папку intruders
                color = (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                cv2.putText(img, "!!! UNKNOWN !!!", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)
                

                # Get the current datetime in the desired format
                current_datetime = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                file_name_path = f"intruders/{current_datetime}.jpg"
                
                cv2.imwrite(file_name_path, img)


            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        #coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 255, 0), "Face", clf)
        coords = draw_boundary(img, faceCascade, 1.1, 10, clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # clf = cv2.face.LBPHFaceRecognizer_create()
    clf = load_model("classifier.keras")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()


b2 = tk.Button(window, text="Detect the face", font=("Courier New", 20), bg="#404040", fg="#000000", command=detect_face)
b2.grid(column=1, row=4)


def generate_dataset():
    if (t1.get() == "" or t2.get() == "" or t3.get() == ""):
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="root",
            database="Authorized_user"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * from my_table")
        myresult = mycursor.fetchall()
        id = 1
        for x in myresult:
            id += 1
        sql = "insert into my_table(id,Name,Age,Address) values(%s,%s,%s,%s)"
        val = (id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # scaling factor=1.3
            # Minimum neighbor = 5

            if faces is ():
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (224, 224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # (50,50) is the origin point from where text is to be written
                # font scale=1
                # thickness=2

                cv2.imshow("Cropped face", face)
                if cv2.waitKey(1) == 13 or int(img_id) == 1000:
                    break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!!')


b3 = tk.Button(window, text="Generate dataset", font=("Courier New", 20), bg="#B6B6B6", fg="#006600", command=generate_dataset)
b3.grid(column=0, row=4)


# кнопка для просмотра неавторизованных пользователей
b4 = tk.Button(window, text="INTRUDERS", font=("Courier New", 20), bg='#F20000', fg="#000000", command=img_navigator.show_image_navigator)
b4.grid(column=1, row=5)



window.geometry("800x300")
window.mainloop()