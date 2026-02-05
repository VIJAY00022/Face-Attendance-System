import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# Load known faces
known_encodings = []
known_names = []

for file in os.listdir("faces"):
    if file.endswith(".npy"):
        encoding = np.load(f"faces/{file}")
        known_encodings.append(encoding)
        known_names.append(file.replace(".npy", ""))

# CSV file create if not exists
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Punch In", "Punch Out"])

cap = cv2.VideoCapture(0)
marked_today = {}

print("Attendance system started... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            key = f"{name}_{date}"

            rows = []
            with open("attendance.csv", "r") as f:
                rows = list(csv.reader(f))

            found = False
            for row in rows:
                if row[0] == name and row[1] == date:
                    found = True
                    if row[3] == "":
                        row[3] = time
                        print(f"👋 {name} Punch Out at {time}")
                    break

            if not found:
                rows.append([name, date, time, ""])
                print(f"✅ {name} Punch In at {time}")

            with open("attendance.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
