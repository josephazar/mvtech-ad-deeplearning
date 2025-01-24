import argparse
import os
import shutil

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=int)

    args = parser.parse_args()

    dataset_path = args.dataset_path

    if not os.path.exists(dataset_path):
        print(f"The directory '{dataset_path}' does not exist.")

    else:
        for cat in os.listdir(dataset_path):

            if os.path.isdir(os.path.join(dataset_path, cat)):
                sub_dir_content = os.listdir(dataset_path, cat)

                for f in sub_dir_content:
                    if f.lower().strip()=='disthresh':
                        os.makedirs(os.path.join(dataset_path, cat, f, "good"), exist_ok=True)
                        for mask in os.path.join(dataset_path, cat, f):
                            shutil.move(os.path.join(dataset_path, cat, f, mask), os.path.join(dataset_path, cat, f, "good", mask))