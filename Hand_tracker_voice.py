import cv2
import mediapipe as mp
from gtts import gTTS
import os

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar TTS (Text to Speech)
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")

# Crear objeto Hand Tracker
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Función para detectar el gesto de "pulgar hacia arriba"
def detect_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_y = thumb_tip.y
    index_y = index_tip.y
    if thumb_y < index_y:  # Si la punta del pulgar está arriba de la punta del índice
        return True
    return False

# Función para detectar el gesto de "pulgar hacia abajo"
def detect_thumb_down(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_y = thumb_tip.y
    index_y = index_tip.y
    if thumb_y > index_y:  # Si la punta del pulgar está abajo de la punta del índice
        return True
    return False

# Función para detectar el gesto de "señal de paz"
def detect_peace(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_y = index_tip.y
    middle_y = middle_tip.y
    if middle_y < index_y:  # Si la punta del dedo medio está arriba de la punta del índice
        return True
    return False

# Función para detectar el gesto de "mano abierta"
def detect_hand_open(hand_landmarks):
    # Comprobamos si los dedos están doblados
    for landmark in [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                     mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                     mp_hands.HandLandmark.PINKY_PIP]:
        if hand_landmarks.landmark[landmark].y < hand_landmarks.landmark[landmark].y:
            return False
    return True

# Función principal
def main():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convertir la imagen a RGB y procesarla con MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los puntos de referencia de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detectar gestos y activar eventos
                if detect_thumb_up(hand_landmarks):
                    speak("Thumbs up detected!")
                elif detect_thumb_down(hand_landmarks):
                    speak("Thumbs down detected!")
                elif detect_peace(hand_landmarks):
                    speak("Peace sign detected!")
                elif detect_hand_open(hand_landmarks):
                    speak("Open hand detected!")

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
