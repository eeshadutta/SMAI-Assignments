from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack(
                (cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape(
        (len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(
            0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape(
        (len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(
            0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


cifar_10_dir = 'cifar-10-batches-py'
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(
    cifar_10_dir)

train_data = np.reshape(train_data, (50000, 3072))
test_data = np.reshape(test_data, (10000, 3072))
print("Train data: ", train_data.shape)
print("Train filenames: ", train_filenames.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test filenames: ", test_filenames.shape)
print("Test labels: ", test_labels.shape)
print("Label names: ", label_names.shape)


def compute_similarity(train_data, train_labels):
    num_images = np.zeros((10))
    mean_images = np.zeros((10, train_data.shape[1]))
    mean_error = np.zeros((10))
    similarity_classes = np.zeros((10, 10))
    train_data_classes = np.empty((10, 5000, train_data.shape[1]))
    eig_vecs_classes = np.empty((10, train_data.shape[1], 20))

    for i in range(train_data.shape[0]):
        for j in range(0, 10):
            if j == train_labels[i]:
                train_data_classes[j][int(num_images[j])] = train_data[i]
                mean_images[j] = mean_images[j] + train_data[i]
                num_images[j] += 1

    for i in range(0, 10):
        mean_images[i] = mean_images[i] / float(num_images[i])

    for i in range(0, 10):
        mean_error[i] = np.mean(
            np.sqrt(np.sum((train_data_classes[i] - mean_images[i])**2, axis=1)))

    for i in range(0, 10):
        data = train_data_classes[i]
        cov = np.cov(data, rowvar=False)
        eig_vals, eig_vecs = LA.eig(cov)
        eig_vecs_classes[i][:, 0:20] = eig_vecs[:, 0:20]

    for i in range(0, 10):
        similarity = []
        ind = list(range(0, 10))
        for j in range(0, 10):
            cl1_compressed = np.matmul(
                train_data_classes[i], eig_vecs_classes[j])
            cl1_reconstructed = np.matmul(
                cl1_compressed, eig_vecs_classes[j].T)
            e1 = np.sum(
                (cl1_reconstructed - train_data_classes[i])**2, axis=1)
            e1 = np.mean(np.sqrt(e1))
            cl2_compressed = np.matmul(
                train_data_classes[j], eig_vecs_classes[i])
            cl2_reconstructed = np.matmul(
                cl2_compressed, eig_vecs_classes[i].T)
            e2 = np.sum(
                (cl2_reconstructed - train_data_classes[j])**2, axis=1)
            e2 = np.mean(np.sqrt(e2))

            err1 = (e1 + mean_error[i]) / (2.0)
            err2 = (e2 + mean_error[j]) / (2.0)

            similarity.append((err1 + err2) / (2.0))

        zipped_pairs = zip(similarity, ind)
        ans = []
        for s, idx in sorted(zipped_pairs):
            if idx != i:
                ans.append(idx)
        # ans = [x for _, x in sorted(zipped_pairs)]
        print("Class", i, ":", ans[0:3])


compute_similarity(train_data, train_labels)
