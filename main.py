from playsound import playsound
import os
import cv2
import mediapipe as mp

sound_FullPath = []
file_list = os.listdir(".\\piano")
for i in range(len(file_list)):
    sound_FullPath.append("piano\\" + file_list[i])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

piano = cv2.imread("pianokey.jpg")
piano = cv2.flip((piano), 1)
piano = cv2.resize(piano, dsize=(650, 200), fx=0.5, fy=0.5)
(h, w) = piano.shape[:2]

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7 ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:

      for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # 손 위치 가져오기
        fingLoc_4 = hand_landmarks.landmark[4] # 엄지
        fingLoc_8 = hand_landmarks.landmark[8] # 검지
        fingLoc_12 = hand_landmarks.landmark[12] # 중지
        fingLoc_16 = hand_landmarks.landmark[16] # 약지
        fingLoc_20  = hand_landmarks.landmark[20] # 새끼
        print(fingLoc_4.z * 1000)
        if(fingLoc_4.z * 1000 > -100):
            playsound(sound_FullPath[0])

        break
    image = cv2.resize(image, dsize=(650, 500), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    image[500-h:500, 0:w] = piano

    cv2.imshow('HandMouse', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

