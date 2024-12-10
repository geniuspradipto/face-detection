from mtcnn import MTCNN
import cv2
import os

detector = MTCNN()

image_path = 'img_590.jpg'  # Path to the input image
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detections = detector.detect_faces(rgb_image)

output_folder = 'detected_faces'
os.makedirs(output_folder, exist_ok=True)

for i, detection in enumerate(detections):
    x, y, width, height = detection['box']
    x, y = max(0, x), max(0, y)
    side_length = max(width, height)
    x_center = x + width // 2
    y_center = y + height // 2
    x_new = max(0, x_center - side_length // 2)
    y_new = max(0, y_center - side_length // 2)
    x_new = min(x_new, image.shape[1] - side_length)
    y_new = min(y_new, image.shape[0] - side_length)
    face = image[y_new:y_new+side_length, x_new:x_new+side_length]
    face_path = os.path.join(output_folder, f'face_{i+1}.jpg')
    cv2.imwrite(face_path, face)
    print(f"Face {i+1} saved to {face_path}")

print(f"Detected {len(detections)} faces. All saved in '{output_folder}' folder.")
