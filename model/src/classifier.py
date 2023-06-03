import math
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC

from model.src import facenet_config as facenet


def train_or_classify(
    mode,
    data_dir,
    model,
    classifier_filename,
    use_split_dataset=False,
    test_data_dir=None,
    batch_size=90,
    image_size=160,
    seed=666,
    min_nrof_images_per_class=20,
    nrof_train_images_per_class=10,
):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            np.random.seed(seed=seed)

            if use_split_dataset:
                dataset_tmp = facenet.get_dataset(data_dir)
                train_set, test_set = split_dataset(
                    dataset_tmp, min_nrof_images_per_class, nrof_train_images_per_class
                )
                if mode == "TRAIN":
                    dataset = train_set
                elif mode == "CLASSIFY":
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (
                    len(cls.image_paths) > 0
                ), "There must be at least one image for each class in the dataset"

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print("Number of classes: %d" % len(dataset))
            print("Number of images: %d" % len(paths))

            # Load the model
            print("Loading feature extraction model")
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "input:0"
            )
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "embeddings:0"
            )
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "phase_train:0"
            )
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print("Calculating features for images")
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict
                )

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            if mode == "TRAIN":
                # Train classifier
                print("Training classifier")
                model = SVC(kernel="linear", probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace("_", " ") for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, "wb") as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif mode == "CLASSIFY":
                # Classify images
                print("Testing classifier")
                with open(classifier_filename_exp, "rb") as infile:
                    (model, class_names) = pickle.load(infile)

                print(
                    'Loaded classifier model from file "%s"' % classifier_filename_exp
                )

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[
                    np.arange(len(best_class_indices)), best_class_indices
                ]

                for i in range(len(best_class_indices)):
                    print(
                        "%4d  %s: %.3f"
                        % (
                            i,
                            class_names[best_class_indices[i]],
                            best_class_probabilities[i],
                        )
                    )

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print("Accuracy: %.3f" % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(
                facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class])
            )
            test_set.append(
                facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:])
            )
    return train_set, test_set


def train_classify():
    mode = "TRAIN"  # Set mode to 'TRAIN' or 'CLASSIFY' based on your requirements
    data_dir = "model/Dataset/FaceData/processed"  # Path to the data directory containing aligned LFW face patches
    model = "model/pretrained/20180402-114759.pb"  # Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file
    classifier_filename = "model/pretrained/facemodel.pkl"  # Classifier model file name as a pickle (.pkl) file
    use_split_dataset = (
        False  # Set to True if the dataset should be split into a training and test set
    )
    test_data_dir = None  # Path to the test data directory (optional)
    batch_size = 1000  # Number of images to process in a batch
    image_size = 160  # Image size (height, width) in pixels
    seed = 666  # Random seed
    min_nrof_images_per_class = (
        20  # Only include classes with at least this number of images in the dataset
    )
    nrof_train_images_per_class = 10  # Use this number of images from each class for training and the rest for testing

    train_or_classify(
        mode=mode,
        data_dir=data_dir,
        model=model,
        classifier_filename=classifier_filename,
        use_split_dataset=use_split_dataset,
        test_data_dir=test_data_dir,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        min_nrof_images_per_class=min_nrof_images_per_class,
        nrof_train_images_per_class=nrof_train_images_per_class,
    )


if __name__ == "__main__":
    train_classify()
