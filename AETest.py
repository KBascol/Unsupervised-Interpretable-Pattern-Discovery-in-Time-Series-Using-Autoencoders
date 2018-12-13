"""
Simple tester
"""

import sys
import argparse
import logging as log

import tensorflow as tf
import numpy as np

from AE import AutoEnc
import dataset_loader_AE as loader

def run_test(argv):
    """ Launch network on chosen arguments """
    nb_channels = 1

    run_name = argv.expe_file.split("/")
    expe_group = run_name[1]
    run_name = run_name[0]


    log.info("Setting session variables")

    with tf.device('/gpu:'+argv.gpu):
        dataset_paths = loader.load_paths(argv.dataset_path)

        tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if argv.from_scratch:
            auto_enc = AutoEnc()
        else:
            auto_enc = AutoEnc(weights_path=argv.weights_path)

        images = tf.placeholder(tf.float32, [None, argv.doc_height, argv.doc_length, nb_channels])

        learning_rate = tf.placeholder(tf.float32)

        log.info("Build Autoencoder")
        auto_enc.build(images, argv.nb_filters, argv.filters_length)

        # Lasso regularization
        #   forces weights to be zero so that remaining weights should be relevent
        lasso = 0
        # Group lasso regularization
        #   forces filters to be empty in order to keep only relevent ones
        grp_lasso = 0
        for variable in tf.trainable_variables():
            if "/filter" in variable.name:
                grp_lasso += tf.reduce_sum(tf.sqrt(tf.reduce_sum(variable**2, axis=(0, 1, 2))))
                lasso += tf.reduce_sum(tf.abs(variable))

        # Kullback-leibler regularization
        #   encourages sparsity in activations
        sum_latent = tf.reduce_sum(auto_enc.latent, axis=(1, 2, 3)) + 10**(-9)
        sum_latent = tf.expand_dims(sum_latent, axis=1)
        sum_latent = tf.expand_dims(sum_latent, axis=2)
        sum_latent = tf.expand_dims(sum_latent, axis=3)

        rho_hat = (auto_enc.latent/sum_latent) + 10**(-9)
        kullback = -tf.reduce_mean(tf.reduce_sum(rho_hat*tf.log(rho_hat),
                                                 axis=(1, 2, 3)))

        # Log loss
        # out_prime = (auto_enc.outputs/(tf.reduce_sum(auto_enc.outputs, axis=(1, 2, 3))
        #                                + 10**(-9))
        #              + 10**(-9))
        # c_func = -tf.reduce_mean(images*tf.log(out_prime))

        # Mean Squared Error
        c_func = tf.reduce_mean(((auto_enc.outputs - images) ** 2))

        cost = c_func + argv.lambdaL*lasso + argv.lambdaGL*grp_lasso + argv.lambdaKL*kullback

        algo = argv.gradient_algorithm.lower()
        if algo == "sgd":
            train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        elif algo == "momentum":
            train = tf.train.MomentumOptimizer(learning_rate, argv.momentum).minimize(cost)
        elif algo == "adam":
            train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        else:
            log.error("Unknown gradient descent algorithm" + argv.gradient_algorithm)
            return None

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("tensorboard/"+run_name, sess.graph)
        tf.summary.scalar(expe_group + "/" + "mse", c_func)
        tf.summary.scalar(expe_group + "/" + "cost", cost)
        summary_op = tf.summary.merge_all()

        if argv.train:
            log.info("Train step")

            for iterat in range(argv.iterations):
                log.info("iteration "+str(iterat+1)+"/"+str(argv.iterations))

                # log.info("Load dataset batch")
                minibatch = loader.load_minibatch(paths_dataset=dataset_paths,
                                                  batch_size=argv.batches_size,
                                                  img_height=argv.doc_height,
                                                  img_width=argv.doc_length,
                                                  nb_channels=nb_channels)


                # log.info("Run network")

                _, summary = sess.run([train, summary_op], feed_dict={images: minibatch,
                                                                      learning_rate: argv.learning_rate})


                writer.add_summary(summary, iterat)

            auto_enc.save_npy(sess, argv.out_weights_path)

        log.info("Test step")
        for variable in tf.trainable_variables():
            if "/filter" in variable.name:
                save_path = variable.name.replace("/", "_")
                save_path = variable.name
                save_path = save_path.split(":")[0]
                save_path = expe_group + "/" + save_path

                var_shape = variable.get_shape().as_list()

                if var_shape[2] == 1:
                    display_var = np.ones(((1, var_shape[0], var_shape[3]*(var_shape[1]+1), 1)))
                    norm_var = sess.run((variable-tf.reduce_min(variable))/(tf.reduce_max(variable)-tf.reduce_min(variable)))

                    for i in range(var_shape[3]):
                        display_var[0, :, i*var_shape[1]+(i+1):(i+1)*var_shape[1]+(i+1), 0] = norm_var[:, :, 0, i]
                    writer.add_summary(sess.run(tf.summary.image(save_path,
                                                                 display_var)))

        example_i = 0
        for image, _ in loader.load_test(paths_dataset=dataset_paths,
                                         img_height=argv.doc_height,
                                         img_width=argv.doc_length,
                                         nb_channels=nb_channels):

            # log.info("Run network")
            out, latent = sess.run([auto_enc.outputs, auto_enc.latent],
                                   feed_dict={images: image})

            example_i += 1

            if example_i == 1:
                writer.add_summary(sess.run(tf.summary.image(expe_group + "/" + "input_" + str(example_i), image)))
                writer.add_summary(sess.run(tf.summary.image(expe_group + "/" + "out_" + str(example_i), out)))
                latent = latent.reshape(latent.shape[2], latent.shape[3])
                latent = np.rot90(latent)
                latent = np.expand_dims(latent, axis=0)
                latent = np.expand_dims(latent, axis=3)
                writer.add_summary(sess.run(tf.summary.image(expe_group + "/" + "latent_" + str(example_i), latent)))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Autoencoder Test.")

    PARSER.add_argument("--train", dest='train', action='store_const',
                        const=True, default=False, help="Launch on training mode.")

    PARSER.add_argument("--scratch", dest='from_scratch', action='store_const',
                        const=True, default=False, help="Enable learning from scratch.")

    PARSER.add_argument("--dataset_path", type=str, default="Example_Dataset/dataset.json",
                        help="Path of the json dataset file.")

    PARSER.add_argument("--doc_length", type=int, default=500,
                        help="Length of a temporal document.")

    PARSER.add_argument("--doc_height", type=int, default=10,
                        help="Height of a temporal document.")

    PARSER.add_argument("--nb_filters", type=int, default=10,
                        help="Number of filters given.")

    PARSER.add_argument("--filters_length", type=int, default=45,
                        help="Length of the given filters.")

    PARSER.add_argument("--weights_path", type=str, default="ae_weights.npy",
                        help="If not scratch, path of the network weights file.")

    PARSER.add_argument("--out_weights_path", type=str, default="trained_ae_weights.npy",
                        help="If train, path where saving the final weights.")

    PARSER.add_argument("--iterations", type=int, default=1000,
                        help="Number of training iterations.")

    PARSER.add_argument("--batches_size", type=int, default=200,
                        help="Number of examples in each batches")

    PARSER.add_argument("--gradient_algorithm", type=str, default="adam",
                        help="Algorithm used for gradient descent (SGD, momentum, ADAM).")

    PARSER.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate used in training.")

    PARSER.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum used in training.")

    PARSER.add_argument("--lambdaGL", type=float, default=1.0,
                        help="Group lasso coefficient on first encode filters and last decode filters.")

    PARSER.add_argument("--lambdaL", type=float, default=0.001,
                        help="Lasso coefficient on first encode filters and last decode filters.")

    PARSER.add_argument("--lambdaKL", type=float, default=0.001,
                        help="Kullback on latent coefficient.")

    PARSER.add_argument("--expe_file", type=str, default="expe/results",
                        help="Used as prefix of the output files path.")

    PARSER.add_argument("--gpu", type=str, default="0",
                        help="GPU-to-be-used index.")

    log.basicConfig(level=log.DEBUG,
                    format="[%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)

    run_test(PARSER.parse_args())
