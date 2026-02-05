import cv2
import face_recognition
import numpy as np
import os

if not os.path.exists("faces"):
    os.makedirs("faces")

name = input("Apna naam enter karo: ")

cap = cv2.VideoCapture(0)
print("Camera ke saamne dekho, 's' dabao save karne ke liye")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ Camera frame nahi aa raha")
        continue

    # 🔥 FIX 1: Agar 4 channel image hai to BGR me lao
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 🔥 FIX 2: Resize (stability ke liye)
    frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)

    # 🔥 FIX 3: Ensure uint8
    frame = frame.astype(np.uint8)

    # 🔥 FIX 4: RGB + contiguous memory
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb, model="hog")
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if len(face_encodings) == 1:
            np.save(f"faces/{name}.npy", face_encodings[0])
            print("✅ Face Registered Successfully")
            break
        else:
            print("⚠️ Sirf ek face hona chahiye")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
