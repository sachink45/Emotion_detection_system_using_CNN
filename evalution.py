import cv2
import numpy as np
from keras.models import model_from_json


# load model configuration from json file
json_file = open('model_json.json', 'r')
load_json_model = json_file.read()
json_file.close()
emotion_model = model_from_json(load_json_model)

# load weights into model
emotion_model.load_weights('emotion_rec_model.h5')
print('model weights has been loaded')

# creating categories of emotion in dictionary
emotion_list = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad", 6:"surprise"}
# accessing camera to test the model

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("E:\\Songs\\Akull - Laal Bindi.mkv")


while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1200,720))
        if not ret:
                break
        face_detect = cv2.CascadeClassifier('D:/har/opencvRepo-Face-Recognition---Haarcascade-main/haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face available on camera
        num_faces = face_detect.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        # slecet each face available in frame and process it
        for (x,y,w,h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0,0,255), 4)
                roi_gray_frame = gray_img[y:y + h, x:x +w]
                selected_cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)


                # prediction

                emotion_prediction = emotion_model.predict(selected_cropped_img)
                maximum_index = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_list[maximum_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
