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


cifar_10_dir = 'files'
train_data, train_filenames, train_labels, test_data,test_filenames,test_labels, label_names = load_cifar_10_data(cifar_10_dir)

train_data = np.reshape(train_data,(50000,3072))
test_data = np.reshape(test_data,(10000,3072))
print("Train data: ", train_data.shape)
print("Train filenames: ", train_filenames.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test filenames: ", test_filenames.shape)
print("Test labels: ", test_labels.shape)
print("Label names: ", label_names.shape)

# print(train_labels)
def compute_similarity_classes(train_data,train_labels):

    means = np.zeros((10,train_data.shape[1]))
    counts = np.zeros((10))
    # cats_err = np.zeros((10))
    mean_errors = np.empty((10))

    cats_similarity = np.empty((10,10))
    
    cats_eig_vecs_top = np.empty((10,train_data.shape[1],20))
    cats_data = np.empty((10,5000,train_data.shape[1]))

    for i in range(train_data.shape[0]):
        for j in range(0,10):
            if j == train_labels[i]:
                cats_data[j][int(counts[j])] = train_data[i]
                counts[j] += 1
                means[j] = means[j]+train_data[i]

    ## Mean images 
    for i in range(10):
        means[i] = means[i]/float(counts[i])
    
    ## Errors represting by mean of each class
    for i in range(10):
        error = np.sum((cats_data[i] - means[i])**2)
        error /= float(means.shape[0])    
        mean_errors[i] = error
    
    print(mean_errors)

    ## Compute 20 top eign vectors for each class
    for i in range(10):
        # crr_mat = cats_data[i]
        # cov_mat = np.cov(crr_mat,rowvar=False)
        # eig_vals,eig_vecs = LA.eig(cov_mat)
        # cats_eig_vecs_top[i][:,0:20] = eig_vecs[:,0:20]
        print(cats_eig_vecs_top[i].shape)
    
    ## For each class pick the 3 closest classes
    for i in range(10):

        class_similarity = []
        indices = list(range(0,10))

        for j in range(10):
            ##compress using other pca
            class_a_comp = np.matmul(cats_data[i],cats_eig_vecs_top[j])
            ## reconstruct
            class_a_recons = np.matmul(class_a_comp,cats_eig_vecs_top[j].T)
            ## compute error
            error1 = np.sum((class_a_recons-cats_data[i])**2)
            error1 /= float(cats_data[i].shape[1])

            class_b_comp = np.matmul(cats_data[j],cats_eig_vecs_top[i])
            ## reconstruct
            class_a_recons = np.matmul(class_a_comp,cats_eig_vecs_top[i].T)
            ## compute error
            error2 = np.sum((class_a_recons-cats_data[j])**2)
            error2 /= float(cats_data[j].shape[1])

            error1 = (error1 + mean_errors[i])/(2.0)
            error2 = (error2 + mean_errors[j])/(2.0)

            error = (error1+error2)/(2.0)

            class_similarity.append(error)

        
        zipped_pairs = zip(class_similarity, indices) 
  
        z = [x for _, x in sorted(zipped_pairs)]

        print(z[1:3])

    
compute_similarity_classes(train_data,train_labels)




