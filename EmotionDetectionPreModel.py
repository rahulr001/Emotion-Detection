import cv2
from deepface import DeepFace
import time
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("cannot open cam")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame,enforce_detection=False, actions=['emotion'])
    print(result[0]['dominant_emotion'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, z, w) in face:
        cv2.rectangle(frame, (x, y), (x+z, y+w), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame ,result[0]['dominant_emotion'],
                (50, 100), font, 3, (0, 0, 255),3)
    cv2.imshow('Demo Video', frame)
    # time.sleep(2)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
