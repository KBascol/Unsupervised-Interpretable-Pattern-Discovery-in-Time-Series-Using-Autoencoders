import os
import re
import logging as log
from random import random

import numpy as np
import simplejson

def load_json(file_path):
    """ Parse a file content for remove unsupported comments and load as JSON """
    with open(file_path) as json_file:
        json_file_content = json_file.read()
        json_file_content = re.sub(r".*//.*", "", json_file_content)
        parsed = simplejson.loads(json_file_content)  # result is now a dict
        return parsed
    raise Exception("Impossible to read JSON file " + file_path)

def write_json(file_path, content):
    """ Parse a file content for remove unsupported comments and load as JSON """
    with open(file_path, 'w') as json_file:
        json_content = simplejson.dumps(content, indent=4 * ' ')
        json_file.write(json_content)
    # raise Exception("Impossible to write JSON file " + file_path)

def read_dir(dir_path):
    """ Return dir_path content if dir exists, display error and return empty list otherwise """

    if os.path.isdir(dir_path):
        return os.listdir(dir_path)

    log.error("No directory with path " + dir_path)
    return []


def load_tdoc(path, height, width, nb_channels):
    '''
    Loads the tdoc file path.
    '''
    infile = open(path, 'r')
    tdoc = np.zeros((height, width, nb_channels))

    instant_i = 0
    for instant in infile:
        if instant_i == width:
            break

        match_com = re.match(r'#', instant)
        if not match_com:
            if len(instant.strip()) != 0:
                pixels = instant.rstrip().split(' ')

                for pix in pixels:
                    pix_height, pix_weigth = pix.split(':')
                    pix_height = int(pix_height)
                    pix_weigth = float(pix_weigth)

                    if pix_height >= height:
                        break

                    for chan in range(nb_channels):
                        tdoc[pix_height, instant_i, chan] = pix_weigth
        instant_i += 1

    infile.close()

    #tdoc = (tdoc-tdoc.min())/(tdoc.max()-tdoc.min())

    return tdoc

def load_paths(dataset_path):
    dataset = load_json(dataset_path)
    return dataset

def load_minibatch(paths_dataset, batch_size, img_height, img_width, nb_channels):
    paths_dataset = paths_dataset["train"]
    nb_examples = len(paths_dataset)

    dataset = np.empty((batch_size, img_height, img_width, nb_channels))


    for example_i in range(batch_size):
        rand_ex_i = np.random.randint(0, nb_examples)

        example_path = paths_dataset[rand_ex_i]
        #log.info("load image " + example_path)

        dataset[example_i] = load_tdoc(path=example_path,
                                       height=img_height,
                                       width=img_width,
                                       nb_channels=nb_channels)

    return dataset

def load_test(paths_dataset, img_height, img_width, nb_channels):
    for example_path in paths_dataset["test"]:
        example = load_tdoc(path=example_path,
                            height=img_height,
                            width=img_width,
                            nb_channels=nb_channels)
        example = np.array([example])
        yield (example, example_path)

def create_json_dataset(dataset_dir, dataset_name, nb_tests):
    tdocs_dir = os.path.join(dataset_dir, "Tdocs")
    tdocs = read_dir(tdocs_dir)

    tdocs_paths = [os.path.join(tdocs_dir, tdoc) for tdoc in tdocs]

    dict_dataset = {"test":tdocs_paths[:nb_tests],
                    "train":tdocs_paths[nb_tests:]}

    write_json(os.path.join(dataset_dir, dataset_name+".json"), dict_dataset)

if __name__ == "__main__":
    create_json_dataset("Datasets/",
                        "Datasets/dataset.json",
                        10)
