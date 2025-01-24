import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('--dataset_path', default='./datasets/AeBAD/AeBAD_S', type=str, help='The path to the original dataset')
    parser.add_argument('--new_path', default='./datasets/preprocessed', type=str, help='The path to the restructured dataset folder')

    config = parser.parse_args()

    dataset_path = config.dataset_path
    new_path = config.new_path

    dataset_name = os.path.basename(dataset_path)

    for subset in os.listdir(dataset_path):
        subset_path = os.path.join(dataset_path, subset)

        if not os.path.isdir(subset_path):
            continue

        for state in os.listdir(subset_path):
            state_path = os.path.join(subset_path, state)

            if not os.path.isdir(state_path):
                continue

            for aug in os.listdir(state_path):
                aug_path = os.path.join(state_path, aug)

                if not os.path.isdir(aug_path):
                    continue

                destination_path = os.path.join(new_path, f'{dataset_name}_{aug}', '0', subset, state)
                os.makedirs(destination_path, exist_ok=True)

                for file_name in os.listdir(aug_path):
                    source_file = os.path.join(aug_path, file_name)

                    if os.path.isfile(source_file):
                        shutil.copy2(source_file, os.path.join(destination_path, file_name))
                        print(f"Copied: {source_file} -> {destination_path}")
