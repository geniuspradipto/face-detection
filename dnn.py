import cv2
import os

# Load the pretrained DNN model
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
config_path = 'deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Input image path
image_path = 'img_342.jpg'
image = cv2.imread(image_path)
(h, w) = image.shape[:2]

# Preprocess the image for the DNN
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the image through the network
net.setInput(blob)
detections = net.forward()

# Output folder to save detected features
output_folder = 'detected_faces_and_parts'
os.makedirs(output_folder, exist_ok=True)

# Loop over detections and process them
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    # Consider detections with a confidence above 0.5
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")
        
        # Ensure bounding boxes are within image dimensions
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w, endX), min(h, endY)
        
        # Crop the detected region
        detected_face = image[startY:endY, startX:endX]
        
        # Save the detected face or partial face
        face_path = os.path.join(output_folder, f'detected_{i+1}.jpg')
        cv2.imwrite(face_path, detected_face)
        print(f"Detection {i+1} saved to {face_path}")

print(f"Detection complete. Results saved in '{output_folder}'.")
