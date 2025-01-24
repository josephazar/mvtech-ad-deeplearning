import os

def change_extension(directory):

    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            old_path = os.path.join(directory, filename)
            new_filename = os.path.splitext(filename)[0] + ".JPG"
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")

directory_path = "./datasets/preprocessed/VisA/candle/ground_truth/bad"
change_extension(directory_path)
