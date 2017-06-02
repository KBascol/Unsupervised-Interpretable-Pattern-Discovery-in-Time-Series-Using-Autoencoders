#!/usr/bin/env python
# coding: utf-8

##############################################################
# Authors  :  Dawood Al Chanti                               #
#             Kevin Bascol                                   #
# Project : Unsupervised Temporal Pattern Discovery in       #
#           Video using Convolutional Auto Encoder           #
##############################################################

'''
Handles the generation of synthetic tdoc
'''
import argparse

PARSER = argparse.ArgumentParser(description="Temporal Document Generator.")

PARSER.add_argument("--gene_img", dest='gene_img', action='store_const',
                    const=True, default=False,
                    help="Also generate png versions of the generated temporal documents.")

PARSER.add_argument("nb_doc", type=int, default=600,
                    help="Number of temporal document to be generated.")

PARSER.add_argument("nb_test_tdoc", type=int, default=200,
                    help="Number of test examples in dataset")

PARSER.add_argument("doc_length", type=int, default=25,
                    help="Length of a temporal document")

PARSER.add_argument("doc_height", type=int, default=25,
                    help="Height of a temporal document")

PARSER.add_argument("font_name", type=str, default="JennaSue.ttf",
                    help="Path of the font file.")

PARSER.add_argument("font_size", type=int, default=20,
                    help="Size of the font used.")

PARSER.add_argument("motifs", type=str, default="egg_eggplant",
                    help="Words used as motifs (seperated with '_').")

PARSER.add_argument("motifs_length", type=int, default=45,
                    help="Maximum length of a motif.")

PARSER.add_argument("nb_motifs_doc", type=int, default=10,
                    help="Number of motifs in each document.")

PARSER.add_argument("min_nb_occ_motifs", type=int, default=3500,
                    help="Minimum number of observations in a motif.")

PARSER.add_argument("max_nb_occ_motifs", type=int, default=4500,
                    help="Maximum number of observations in a motif.")

PARSER.add_argument("noise", type=float, default=0.33,
                    help="""Proportion of the mean number of obervations in the generated documents
                            added as salt-and-pepper noise.""")

PARSER.add_argument("repository", type=str, default=".",
                    help="Repository where generate the dataset.")

ARGV = PARSER.parse_args()

import os

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.cm as cm
from matplotlib import pyplot as plt

import dataset_loader_AE as loader


def get_size(string, font):
    '''
    Returns the given text size in pixels
    '''
    test_img = Image.new('L', (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    return test_draw.textsize(string, font)

# Convert String to Matrix
def string_to_matrix(string, repo):
    '''
    Returns the pixel matrix corresponding to the given string
    '''
    # Define the Text Color and the Background
    color_text = "White"
    color_background = "Black"

    #Define the image font and resize the nword in a rectangle that suit it
    font = ImageFont.truetype(ARGV.font_name, ARGV.font_size)
    str_l, str_h = get_size(string, font)
    pos_l = max(1, (ARGV.motifs_length-str_l)//2)
    pos_h = max(1, (ARGV.doc_height-str_h)//2)
    img = Image.new('L', (ARGV.motifs_length, ARGV.doc_height), color_background)
    drawing = ImageDraw.Draw(img)
    drawing.text((pos_l, pos_h), string, fill=color_text, font=font)

    img.save(os.path.join(repo, "motif_" + string + ".png"))

    motif = np.asarray(img, np.float32)   # Motif Matrix

    return motif


def motif_matrix_to_norm_vector(motif):
    '''
    Converts Motif to vector in form of Distribution.
    '''
    motif_as_vector = motif.reshape((ARGV.motifs_length*ARGV.doc_height))
    # normalizing as a distribution
    motif_as_vector = motif_as_vector / motif_as_vector.sum()
    motif_as_vector *= .999999

    return motif_as_vector

def string_to_norm_vector(string, repo):
    motif = string_to_matrix(string, repo)

    return motif_matrix_to_norm_vector(motif)

def creat_tdoc_from_motif(motifs, motifs_str):
    '''
    Returns a matrix representing a tdoc
    '''
    tdoc = np.zeros((ARGV.doc_length, ARGV.doc_height), dtype=np.uint8)

    for _ in range(ARGV.nb_motifs_doc):
        # Random-motif's weights that we will draw from
        index_motif = np.random.randint(len(motifs))
        motif = motifs[index_motif]

        start_time = np.random.randint(ARGV.doc_length - ARGV.motifs_length)
        nb_obs = np.random.randint(ARGV.max_nb_occ_motifs - ARGV.min_nb_occ_motifs)
        nb_obs += ARGV.min_nb_occ_motifs
        # taking ito account the size of the motif to avoid too low intensity in long motifs
        nb_obs *= len(motifs_str[index_motif])//5+1

        for _ in range(nb_obs):
            # Draw samples from a multinomial distribution.
            pixel_position = np.argmax(np.random.multinomial(1, motif))

            h_pos = pixel_position // ARGV.motifs_length
            t_pos = pixel_position % ARGV.motifs_length

            time = t_pos + start_time
            tdoc[time][h_pos] += 1

    noise_intensity = (ARGV.noise
                       *((ARGV.max_nb_occ_motifs+ARGV.min_nb_occ_motifs)//2)
                       *ARGV.nb_motifs_doc)
    noise_intensity_max = 0.1*tdoc.max()
    while noise_intensity > 0:
        t_noise = np.random.randint(ARGV.doc_length)
        w_noise = np.random.randint(ARGV.doc_height)

        v_noise = np.random.randint(int(noise_intensity_max))

        tdoc[t_noise][w_noise] += v_noise
        noise_intensity -= v_noise
    return tdoc


def generate():
    repo_tdocs = os.path.join(ARGV.repository, "Tdocs")
    repo_motifs = os.path.join(ARGV.repository, "motifs")

    os.makedirs(repo_tdocs)
    os.makedirs(repo_motifs)

    if ARGV.gene_img:
        repo_imgs = os.path.join(ARGV.repository, "images")
        os.makedirs(repo_imgs)


    motifs_str = ARGV.motifs.split("_")

    # Convert the big normalized matrix into a vector
    motifs_norm_vect = [string_to_norm_vector(motif, repo_motifs) for motif in motifs_str]


    for i in range(0, ARGV.nb_doc):
        temp_doc = creat_tdoc_from_motif(motifs_norm_vect, motifs_str)

        if ARGV.gene_img:
            plt.imshow(np.transpose(temp_doc), cmap=cm.Greys_r)
            plt.savefig(os.path.join(repo_imgs, "Tdoc_"+str((i+1))+".png"))

        output = open(os.path.join(repo_tdocs, "Tdoc_"+str((i+1))+".txt"), 'w')
        for lenght in range(ARGV.doc_length):
            for height in range(ARGV.doc_height):
                if temp_doc[lenght][height] != 0.0:
                    output.write("%s:%s " %(height, temp_doc[lenght][height]))
            output.write("\n")
        output.close()


# Generate documents
if __name__ == '__main__':
    if os.path.isdir(ARGV.repository):
        generate()

        if ARGV.nb_test_tdoc >= 0:
            loader.create_json_dataset(ARGV.repository,
                                       "dataset",
                                       ARGV.nb_test_tdoc)
        else:
            print("[ERROR] Cannot have " + str(ARGV.nb_test_tdoc) + " tdocs in test set.")
    else:
        print("[ERROR] The repository " + ARGV.repository + " doesn't exist.")
