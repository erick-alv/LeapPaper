import numpy as np
import tensorflow.compat.v2 as tf
from ownimplementations.convolutional_vae import CVAE
import time
from ownimplementations.npy_image_reader import store_image_array_at
from ownimplementations.conv_vae_keras import CVAE_Keras
from ownimplementations.conv_vae_keras import compute_apply_gradients, compute_loss
import argparse

batch_size=256
filename = '/home/erick/log_leap/pnr/05-20-generate-vae-dataset-local/05-20-generate-vae-dataset-local_2020_05_20_22_58_16_id000--s94822/vae_dataset.npy'
num_epochs = 1000

def train(args):
    # LOADING THE DATA
    dataset = np.load(filename, allow_pickle=True).item()
    for k in dataset.keys():
        dataset[k] = dataset[k][:10000, :]
    N = 10000
    n = int(N * 0.9)
    # shaping as image 84x84x3
    dataset['obs'] = np.array([transform_to_rbgarry(arr) for arr in dataset['obs']]).astype(np.float32)
    dataset['next_obs'] = np.array([transform_to_rbgarry(arr) for arr in dataset['next_obs']]).astype(np.float32)
    # Normalizing the images to the range of [0., 1.]
    dataset['obs'] /= 255.
    dataset['next_obs'] /= 255.

    # create data_sets
    train_data = {}
    test_data = {}
    for k in dataset.keys():
        train_data[k] = dataset[k][:n, :]
        test_data[k] = dataset[k][n:, :]

    #image is flattened
    conv_vae = CVAE(lr=0.00001, input_channels=3, action_size=2, representation_size=16, imsize=84)

    train_loss = []
    ev_loss = []
    for epoch in range(num_epochs):
        #train
        epoch_train_losses = []
        batches = make_batches(len(train_data['next_obs']), batch_size=batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            im_batch = train_data['next_obs'][batch_start:batch_end].copy()
            z, mu, log_sigma, _, _, loss = conv_vae.train(im_batch)
            '''print("z = ", z[0])
            print("mu = ", mu[0])
            print("log_sigma = ", log_sigma[0])'''
            '''print("losses: rex: ", None," kl: ", None," total: ", loss)'''
            epoch_train_losses.append(loss)
        train_loss.append(np.array(epoch_train_losses).mean())
        #evaluate
        if epoch % 5 == 0:
            epoch_ev_loss = []
            batches = make_batches(len(test_data['next_obs']), batch_size=batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                im_batch = test_data['next_obs'][batch_start:batch_end].copy()
                assert np.array(im_batch).max() <= 1.0
                loss = conv_vae.evaluate(im_batch)
                epoch_ev_loss.append(loss)
            ev_loss.append(np.array(epoch_ev_loss).mean())
            print('Epoch ', epoch, ':')
            print('epoch train_loss: ', np.array(epoch_train_losses).mean())
            print('epoch  evaluation_loss: ', np.array(epoch_ev_loss).mean())
            print('_______________________________________________')
            save_model_file = args.dirpath + "cvae"
            conv_vae.saver.save(conv_vae.sess, save_model_file, global_step=epoch)
    #saves last model
    save_model_file = args.dirpath + "cvae_last"
    conv_vae.saver.save(conv_vae.sess, save_model_file)
    #last log
    print('overall train_loss: ', np.array(train_loss).mean())
    print('overall evaluation_loss: ', np.array(ev_loss).mean())

def transform_to_rbgarry(flattened_array):
    img_array = flattened_array.reshape(3, 84, 84).transpose((1, 2, 0))
    #img_array = img_array[::, :, ::-1]
    return img_array

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

def prove_store(args):
    z = np.random.normal(size=16)
    conv_vae1 = CVAE(lr=0.001, input_channels=3, action_size=2, representation_size=16, imsize=84)
    d1 = conv_vae1.decode(z)
    print(d1)
    d11 = conv_vae1.decode(z)
    print(d11 == d1)
    conv_vae2 = CVAE(lr=0.001, input_channels=3, action_size=2, representation_size=16, imsize=84)
    d2 = conv_vae2.decode(z)
    print(d2)
    print(d1 == d2)
    cvae_file = args.dirpath + "cvae1"
    conv_vae1.saver.save(conv_vae1.sess, cvae_file)
    cvae_file = args.dirpath + "cvae2"
    conv_vae2.saver.save(conv_vae2.sess, cvae_file)
    return d2, d11, d2, z

def prove_load(d2, d11, d1, z, args):
    restore_path = args.dirpath+ "cvae1"
    conv_vae1 = CVAE(lr=0.001, input_channels=3, action_size=2, representation_size=16, imsize=84, restore_path=restore_path)
    d1n = conv_vae1.decode(z)
    print(d1 == d1n)
    d11n = conv_vae1.decode(z)
    print(d11 == d11n)
    restore_path = args.dirpath + "cvae2"
    conv_vae2 = CVAE(lr=0.001, input_channels=3, action_size=2, representation_size=16, imsize=84, restore_path=restore_path)
    d2n = conv_vae2.decode(z)
    print(d2 == d2n)

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirpath', help='path to to directory save/load the model', type=str, default='')
    args = parser.parse_args()
    #d2, d11, d1, z = prove_store(args)
    #prove_load(d2, d11, d1, z, args)
    train(args)
