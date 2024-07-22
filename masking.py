import cv2
import dlib
import os
import numpy as np
import argparse

def mask_eyes_in_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Get the landmarks/parts for the face
        landmarks = predictor(gray, face)

        # Get the coordinates for the left and right eyes
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]

        # Draw a black rectangle over each eye
        left_eye_points = [(point.x, point.y) for point in left_eye]
        right_eye_points = [(point.x, point.y) for point in right_eye]

        # Calculate bounding boxes for eyes
        left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv2.boundingRect(np.array(left_eye_points))
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv2.boundingRect(np.array(right_eye_points))

        left_eye_x = left_eye_x - (left_eye_w // 2)
        left_eye_y = left_eye_y - (left_eye_h)
        left_eye_w = left_eye_w * 2
        left_eye_h = left_eye_h * 3

        right_eye_x = right_eye_x - (right_eye_w // 2)
        right_eye_y = right_eye_y - (right_eye_h)
        right_eye_w = right_eye_w * 2
        right_eye_h = right_eye_h * 3
        
        # Draw black rectangles over the eyes
        cv2.rectangle(image, (left_eye_x, left_eye_y), (left_eye_x + left_eye_w, left_eye_y + left_eye_h), (0, 0, 0), -1)
        cv2.rectangle(image, (right_eye_x, right_eye_y), (right_eye_x + right_eye_w, right_eye_y + right_eye_h), (0, 0, 0), -1)

    # Save the result
    cv2.imwrite(output_path, image)

def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_dir)
                output_file_path = os.path.join(output_dir, relative_path)

                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # Mask the eyes in the image and save the result
                mask_eyes_in_image(input_file_path, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a directory to mask eyes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    
    args = parser.parse_args()
    
    input_dirs = ['train/fake', 'train/real']
    output_dirs = ['train/fake', 'train/real']

    for input_subdir, output_subdir in zip(input_dirs, output_dirs):
        process_directory(os.path.join(args.input_dir, input_subdir), os.path.join(args.output_dir, output_subdir))
