import argparse
import glob
import os
import pathlib
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    data=[source + '/' + file for file in os.listdir(source)]
    np.random.shuffle(data)
    
    # spliting files
    train_files, val_file, test_file = np.split(data, [int(.75*len(data)), int(.9*len(data))])

    # create dirs and move data files into them
    train = pathlib.Path(destination) / 'train'
    train.mkdir(parents=True, exist_ok=True)
    
    for file in train_files:
        shutil.move(file, train)
    
    val = pathlib.Path(destination) / 'val'
    val.mkdir(parents=True, exist_ok=True)
    
    for file in val_file:
        shutil.move(file, val)
    
    test = pathlib.Path(destination) / 'test'
    test.mkdir(parents=True, exist_ok=True)

    for file in test_file:
        shutil.move(file, test) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)