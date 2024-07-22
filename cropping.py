import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
from tqdm import tqdm
import argparse
from dataset.loader import normalize_data
from model.config import load_config
from model.genconvit import GenConViT
from decord import VideoReader, cpu

device = "cuda" if torch.cuda.is_available() else "cpu"

# cropping faces from images and resizing them to 224x224
def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(
                    face_image, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)

def preprocess_and_save(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over train/test/valid folders and their subfolders
    for folder_name in os.listdir(input_folder):
        input_subfolder = os.path.join(input_folder, folder_name)
        output_subfolder = os.path.join(output_folder, folder_name)
        
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        
        for label in ['fake', 'real']:
            input_label_folder = os.path.join(input_subfolder, label)
            output_label_folder = os.path.join(output_subfolder, label)
            
            if not os.path.exists(output_label_folder):
                os.makedirs(output_label_folder)
            
            # Process each image in the input label folder
            for filename in os.listdir(input_label_folder):
                input_image_path = os.path.join(input_label_folder, filename)
                output_image_path = os.path.join(output_label_folder, filename)
                
                # Preprocess and save the image
                preprocess_and_save_image(input_image_path, output_image_path)

def preprocess_and_save_image(input_image_path, output_image_path):
    # Read image using OpenCV
    image = cv2.imread(input_image_path)
    face, count = face_rec([image])
    for frame in face:
        print("Saving to path: ", output_image_path)
        cv2.imwrite(output_image_path, frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images and save them.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    
    args = parser.parse_args()
    
    preprocess_and_save(args.input_folder, args.output_folder)
