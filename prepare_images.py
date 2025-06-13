import os
import cv2

image_dir = "C:/Users/ahmed/Desktop/face_detector/Face-Counter/face_images"  

def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load {filename}")
                continue
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"Preprocessed {filename}")

preprocess_images(image_dir, "C:/Users/ahmed/Desktop/face_detector/Face-Counter/preprocessed_face_images")

print("Image preprocessing complete.")