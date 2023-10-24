# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:52:56 2023

@author: Terrance Williams
@description: Takes in the path of an image dataset and produces text files that split the paths into train, test, and val.
Inspired by Hiwonder's 'text_gen.py' script.
"""

from pathlib import Path
import argparse
import random


# %% Function Defs

def split(image_list: list, shuffle:bool=False) -> tuple:
    """
    Splits the provided dataset into "train", "test", and "val"

    Parameters
    ----------
    dataset : Path
        Path to the dataset
    shuffle : bool, optional
        Determines if the output is shuffled. The default is False.

    Returns
    -------
    tuple
        test, val, train
    """

    n_total = len(image_list)
    offset_test = int(0.05 * n_total)
    offset_val = int(0.20 * n_total)

    if n_total == 0:
        raise ValueError("Image list is empty.")
    if shuffle:
        random.shuffle(image_list)

    # Split the image paths
    test = image_list[:offset_test]
    val = image_list[offset_test:offset_val]
    train = image_list[offset_val:]

    return test, val, train


def generate_test_and_val(data_path: Path):

    image_list = ["".join([str(x), '\n']) for x in data_path.resolve().glob('*.jpg')]

    test, val, train = split(image_list, shuffle=True)
    label_path = data_path / 'label_data'
    if not label_path.exists():
        label_path.mkdir()

    # Generate TXT files
    with open((label_path / 'all.txt'), 'w') as f:
        for file_path in image_list:
            f.write(file_path)

    with open((label_path / 'train.txt'), 'w') as f:
        for file_path in train:
            f.write(file_path)

    with open((label_path / 'val.txt'), 'w') as f:
        for file_path in val:
            f.write(file_path)

    with open((label_path / 'test.txt'), 'w') as f:
        for file_path in test:
            f.write(file_path)

# %% Command-Line

parser = argparse.ArgumentParser()
parser.add_argument('src',
                    default='./',
                    help="Path to dataset source. Defaults to current directory.")


# %% MAIN
if __name__ == '__main__':
    args = parser.parse_args()
    data_path = Path(args.src)
    generate_test_and_val(data_path)
