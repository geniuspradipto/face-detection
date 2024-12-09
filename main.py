import cv2
import os

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Input image path
image_path = 'img_342.jpg'  # Replace with your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Output folder for detected features
output_folder = 'detected_faces_and_eyes'
os.makedirs(output_folder, exist_ok=True)

# Function to save detected regions
def save_region(roi, label, index):
    path = os.path.join(output_folder, f'{label}_{index}.jpg')
    cv2.imwrite(path, roi)
    print(f"{label} {index} saved to {path}")

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Process detected faces
for i, (x, y, w, h) in enumerate(faces):
    face_roi = image[y:y+h, x:x+w]
    save_region(face_roi, "face", i + 1)

# If no faces are detected, attempt to detect eyes
if len(faces) == 0:
    print("No full faces detected. Searching for eyes...")
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
    
    for i, (x, y, w, h) in enumerate(eyes):
        eye_roi = image[y:y+h, x:x+w]
        save_region(eye_roi, "eye", i + 1)

print(f"Detection complete. Results saved in '{output_folder}'.")
