from mtcnn import MTCNN
import cv2
import os
import numpy as np

def nearest_power_of_two(size):
    """
    Find the nearest power of 2 greater than or equal to the given size.
    """
    return 2 ** int(np.ceil(np.log2(size)))

# Initialize MTCNN detector
detector = MTCNN()

# Input image path
image_path = 'img_590.jpg'
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces in the image
detections = detector.detect_faces(rgb_image)

# Output folder for detected faces
output_folder = 'detected_faces'
os.makedirs(output_folder, exist_ok=True)

for i, detection in enumerate(detections):
    x, y, width, height = detection['box']
    x, y = max(0, x), max(0, y)
    
    # Crop the face with equal width and height
    side_length = max(width, height)
    x_center = x + width // 2
    y_center = y + height // 2
    x_new = max(0, x_center - side_length // 2)
    y_new = max(0, y_center - side_length // 2)
    x_new = min(x_new, image.shape[1] - side_length)
    y_new = min(y_new, image.shape[0] - side_length)
    face = image[y_new:y_new+side_length, x_new:x_new+side_length]
    
    # Determine the nearest power of 2 size
    target_size = nearest_power_of_two(max(face.shape[:2]))
    
    # Resize the face to the nearest power of 2
    resized_face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Save the resized face
    face_path = os.path.join(output_folder, f'face_{i+1}.jpg')
    cv2.imwrite(face_path, resized_face)
    print(f"Face {i+1} saved to {face_path} with size {target_size}x{target_size}")

print(f"Detected {len(detections)} faces. All resized to 2^n dimensions and saved in '{output_folder}' folder.")
