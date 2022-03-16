import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

# Binary identification for figures cat/no-cat
# 
def load_dataset():
    '''
    Load datasets of train and test from h5py files
    
    Dataset source: 
    kaggle datasets download -d muhammeddalkran/catvnoncat

    Return:
    train_set_x_orig -- train dataset inputs
    train_set_y_orig -- train dataset labels
    test_set_x_orig -- test dataset inputs
    test_set_y_orig -- test dataset labels
    classes -- 

    '''
    with h5py.File('/home/zzhao/Documents/Code_ML/dataset/catvnoncat/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('/home/zzhao/Documents/Code_ML/dataset/catvnoncat/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))
    
    return s

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print(classes)

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
#print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))



# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

