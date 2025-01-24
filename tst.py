import argparse
import os
import shutil

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)

    args = parser.parse_args()

    dataset_path = args.dataset_path

    if not os.path.exists(dataset_path):
        print(f"The directory '{dataset_path}' does not exist.")
    else:
        for cat in os.listdir(dataset_path):
            cat_path = os.path.join(dataset_path, cat)

            if os.path.isdir(cat_path):
                sub_dir_content = os.listdir(cat_path)

                for f in sub_dir_content:
                    sub_dir_path = os.path.join(cat_path, f)

                    if f.lower().strip() == 'disthresh' and os.path.isdir(sub_dir_path):
                        good_dir_path = os.path.join(sub_dir_path, "good")

                        # Create "good" directory if it doesn't exist
                        os.makedirs(good_dir_path, exist_ok=True)

                        # Move all files in "disthresh" to the "good" directory
                        for mask in os.listdir(sub_dir_path):
                            mask_path = os.path.join(sub_dir_path, mask)

                            # Ensure it's a file before moving
                            if os.path.isfile(mask_path):
                                shutil.move(mask_path, os.path.join(good_dir_path, mask))
