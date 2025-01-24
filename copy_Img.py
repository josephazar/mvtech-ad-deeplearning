import os
import shutil

def copy_and_rename_images(source_folder, destination_folder):
    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all .png files in the source folder
    png_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') and not f.startswith('.')]
    print(len(png_files))
    # Copy and rename each file
    i=len(os.listdir(destination_folder))+1
    print(i)
    for file_name in png_files:
        src_file = os.path.join(source_folder,file_name)
        dest_file = os.path.join(destination_folder, f"{i}.jpg")
        shutil.copy(src_file, dest_file)
        print(f"Copied and renamed: {src_file} -> {dest_file}")
        i=i+1

# Example usage
source_folder = "/home/Dev-YoussefH/Desktop/UFC-Research/datasets/Original_Format/AeBAD/AeBAD/AeBAD_V/train/good/video4_train"  # Replace with the path to your source folder
destination_folder = "/home/Dev-YoussefH/Desktop/UFC-Research/datasets/Modified_Format/no_pixel_annotations/AeBAD_V-EvalVideo1/0/train/good"  # Replace with the path to your destination folder

copy_and_rename_images(source_folder, destination_folder)
