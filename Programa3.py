import cv2

captura = cv2.VideoCapture(0)

ret, frame = captura.read()

if ret:
    cv2.imwrite('C:/Users/JUDITHDELCARMENPADRO/Documents/python/Captura2.jpg', frame)

captura.release()