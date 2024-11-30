import cv2

# Iniciar la captura de video desde la cámara (0 es el índice de la cámara por defecto)
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar Canny para detectar bordes
    canny = cv2.Canny(gray, 10, 150)

    # Dilatar y erodir para mejorar los bordes detectados
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)

    # Encontrar los contornos en la imagen
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procesar cada contorno detectado
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        # Clasificar las figuras según el número de vértices
        if len(approx) == 3:
            cv2.putText(frame, 'Triangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if aspect_ratio == 1:
                cv2.putText(frame, 'Cuadrado', (x, y - 5), 1, 1, (0, 255, 0), 1)
            else:
                cv2.putText(frame, 'Rectangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 5:
            cv2.putText(frame, 'Pentagono', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 6:
            cv2.putText(frame, 'Hexagono', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) > 10:
            cv2.putText(frame, 'Circulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        # Dibujar los contornos en la imagen
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    # Mostrar la imagen con las figuras reconocidas
    cv2.imshow('Figura Geometrica', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
