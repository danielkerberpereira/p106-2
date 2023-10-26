import cv2
import numpy as np

cap = cv2.VideoCapture('walking.avi')

# Crie nosso classificador de corpos
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicie a captura de vídeo para o arquivo de vídeo


# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),2)
        roi_color= frame[y:y+h, x:x+h]
        cv2.imshow("a",roi_color)
    if cv2.waitKey(1) == 32:
        break
cap.release()
cv2.destroyAllWindows()
