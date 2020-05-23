import numpy as np
import tensorflow as tf
from ownimplementations.convolutional_vae import CVAE

def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]

batch_size=128
filename = '/home/erick/log_leap/pnr/05-20-generate-vae-dataset-local/05-20-generate-vae-dataset-local_2020_05_20_22_58_16_id000--s94822/vae_dataset.npy'
num_epochs = 50
def train():
    dataset = np.load(filename, allow_pickle=True).item()
    N = dataset['obs'].shape[0]
    for k in dataset.keys():
        dataset[k] = dataset[k][:10000, :]
    N = 10000
    n = int(N * 0.9)
    '''shaping as image'''
    dataset['obs'] = to_rgb_array_all(N, dataset['obs'])
    dataset['next_obs'] = to_rgb_array_all(N, dataset['obs'])
    store_image_array_at(dataset['next_obs'][0], '/home/erick/log_leap/own_npr/', img_name='train_test')

    train_data = {}
    test_data = {}
    for k in dataset.keys():
        train_data[k] = dataset[k][:n, :]
        test_data[k] = dataset[k][n:, :]

    #image is flattened
    conv_vae = CVAE(lr=0.001, input_channels=3, num_filters=None, action_size=2, representation_size=16, imsize=84)

    for epoch in num_epochs:
        #train
        vae_train_losses = []
        batches = make_batches(len(train_data), batch_size=batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            im_batch = train_data[batch_start:batch_end]['next_obs']
            loss = conv_vae.train(im_batch)
            vae_train_losses.append(loss)
        #evaluate
        vae_ev_losses = []
        batches = make_batches(len(test_data), batch_size=batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            im_batch = test_data[batch_start:batch_end]['next_obs']
            loss = conv_vae.evaluate(im_batch)
            vae_ev_losses.append(loss)
        print('Epoch ', epoch, ':')
        print('train_loss: ', np.array(vae_train_losses).cumsum()[-1])
        print('evaluation_loss: ', np.array(vae_ev_losses).cumsum()[-1])
        print('_______________________________________________')

def transform_to_rbgarry(flattened_array):
    img_array = flattened_array.reshape(3, 84, 84).transpose((1, 2, 0))
    #img_array = img_array[::, :, ::-1]
    return img_array
def to_rgb_array_all(total, array):
    n = np.array([transform_to_rbgarry(arr) for arr in array])
    store_image_array_at(n[0], '/home/erick/log_leap/own_npr/', img_name='hope')
    assert isinstance(array, np.builtins.bytearray)
    img_array = array.reshape(total, 84, 84, 3)
    #img_array = img_array.transpose(0, 2, 3, 1)
    return img_array

from ownimplementations.npy_image_reader import store_image_array_at
def store_images():
    dataset = np.load(filename, allow_pickle=True).item()
    N = dataset['obs'].shape[0]
    '''shaping as image'''
    img = transform_to_rbgarry(dataset['next_obs'][0])
    store_image_array_at(img, '/home/erick/log_leap/own_npr/', img_name='first')



if __name__=='__main__':
    train()