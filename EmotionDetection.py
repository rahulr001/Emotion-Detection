import cv2
import numpy as np
from tensorflow import keras

path = 'haarcascade_frontalface_default.xml'

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bg = (255, 255, 255)
img = np.zeros((500, 500))
text = ''

model = keras.models.load_model('new_trained_model.h5')

(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bg, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 0), 2)
        face = faceCascade.detectMultiScale(roi_gray)

        # if len(face) == 0:
        #     print("no face detected")
        # else:
        # for (ex, ey, ew, eh) in face:
        face_roi = roi_color[y:y + h, x:x + w]

        final_image = cv2.resize(face_roi, (244, 244))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0
        font = cv2.FONT_HERSHEY_DUPLEX

        prediction = model.predict(final_image)

        font_scale = 1.5
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

        if prediction == 'angry':
            status = "Angry"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))


        elif prediction == 'disgusted':
            status = "Disgusted"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))

        elif prediction == 'fearful':
            status = "Fearful"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))

        elif prediction == 'happy':
            status = "Happy"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))

        elif prediction == 'neutral':
            status = "Neutral"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))

        elif prediction == 'sad':
            status = "Sad"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))

        elif prediction == 'surprised':
            status = "Srprised"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 255))
    cv2.imshow('Demo Video', frame)
        # time.sleep(2)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
