# Unsupervised Interpretable Pattern Discovery in Time Series Using Autoencoders
[Link to pdf version of the article](https://hal.archives-ouvertes.fr/hal-01374576/file/sspr_kevin16.pdf)


## Launch the method

usage: python3 AETest.py <arguments>

Mandatory arguments:
*  dataset_path
   * Path of the json dataset file.
*  doc_length
   * Length of a temporal document.
*  doc_height
   * Height of a temporal document.
*  nb_filters
   * Number of filters given.
* filters_length
   * Length of the given filters.
*  weights_path
   * Path of the network weigths file.
*  iterations
   * Number of training iterations.
*  batches_size
   * Number of examples in each batches
*  gradient_algorithm
   * Algorithm used for gradient descent (SGD, momentum, ADAM).
* learning_rate
   * Learning rate used in training.
* momentum
   * Momentum used in training.
* lambdaGL
   * Group lasso coefficient.
*  lambdaL
   * lasso coefficient
*  lambdaKL
   * Kullback on latent coefficient.
*  expe_file
   * Used as prefix of the output files path (format: <run name>/<test detail>).
*  gpu
   * GPU-to-be-used index.

optional arguments:
*  -h, --help
   * show help
*  --train
   * Launch on training mode.
*  --scratch
   * Enable learning from scratch.



The temporal documents are under the form of simple text files, the json dataset file contain a dictionary with the keys "train" and "test" corresponding to lists of path of document (see given example).

When there is several GPUs on the machine, launch "CUDA_VISIBLE_DEVICES=<index of the selected GPU> python3 AETest.py <arguments>" to have a correct memory allocation.


## Generate synthetic data

usage: python3 GenerateTemporalDocument.py <arguments>

Mandatory arguments:
*  nb_doc
   * Number of temporal document to be generated.
*  nb_test_tdoc
   * Number of test examples in dataset
*  doc_length
   * Length of a temporal document
*  doc_height
   * Height of a temporal document
*  font_name
   * Path of the font file.
*  font_size
   * Size of the font used.
*  motifs
   * Words used as motifs (seperated with '_').
*  motifs_length
   * Maximum length of a motif.
*  nb_motifs_doc
   * Number of motifs in each document.
*  min_nb_occ_motifs
   * Minimum number of observations in a motif.
*  max_nb_occ_motifs
   * Maximum number of observations in a motif.
*  noise
   * Proportion of the mean number of obervations in the generated documents added as salt-and-pepper noise.
*  repository
   * Repository where generate the dataset.

optional arguments:
*  -h, --help
   * show help
*  --gene_img
   * Also generate png versions of the generated temporal documents.
