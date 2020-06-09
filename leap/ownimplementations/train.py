import numpy as np
from ownimplementations.convolutional_vae import CVAE
import time
from ownimplementations.npy_image_reader import store_image_array_at
from ownimplementations.conv_vae_keras import CVAE_Keras
import argparse

import pickle
from railrl.torch.vae.conv_vae2 import ConvVAE2
from railrl.torch import pytorch_util as ptu

batch_size=128
filename = '/home/erick/log_leap/pnr/05-20-generate-vae-dataset-local/05-20-generate-vae-dataset-local_2020_05_20_22_58_16_id000--s94822/vae_dataset.npy'
num_epochs = 200

def train(args):
    # LOADING THE DATA
    dataset = np.load(filename, allow_pickle=True).item()
    N = len(dataset['obs'])
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

    #conv_vae = CVAE(lr=0.001, input_channels=3, representation_size=16, imsize=84)
    conv_vae = CVAE_Keras(lr=0.001, input_channels=3, representation_size=16, imsize=84)
    train_rec_loss = []
    train_kl_loss = []
    train_loss = []
    ev_rec_loss = []
    ev_kl_loss = []
    ev_loss = []
    for epoch in range(num_epochs):
        #train
        epoch_train_rec_losses = []
        epoch_train_kl_losses = []
        epoch_train_losses = []
        batches = make_batches(train_data['next_obs'], batch_size=batch_size)
        for batch_index, im_batch in enumerate(batches):
            rec_loss, kl_loss, loss = conv_vae.train(im_batch.copy())
            epoch_train_rec_losses.append(rec_loss)
            epoch_train_kl_losses.append(kl_loss)
            epoch_train_losses.append(loss)
        train_rec_loss.append(np.array(epoch_train_rec_losses).mean())
        train_kl_loss.append(np.array(epoch_train_kl_losses).mean())
        train_loss.append(np.array(epoch_train_losses).mean())
        #evaluate
        if epoch % 50 == 0:
            epoch_ev_rec_loss = []
            epoch_ev_kl_loss = []
            epoch_ev_loss = []
            batches = make_batches(test_data['next_obs'], batch_size=batch_size)
            for batch_index, im_batch in enumerate(batches):
                rec_loss, kl_loss, loss = conv_vae.evaluate(im_batch.copy())
                epoch_ev_rec_loss.append(rec_loss)
                epoch_ev_kl_loss.append(kl_loss)
                epoch_ev_loss.append(loss)
            ev_loss.append(np.array(epoch_ev_loss).mean())
            ev_rec_loss.append(np.array(epoch_ev_rec_loss).mean())
            ev_kl_loss.append(np.array(epoch_ev_kl_loss).mean())
            print('Epoch ', epoch, ':')
            print('epoch train_rec_loss: ', np.array(epoch_train_rec_losses).mean())
            print('epoch train_kl_loss: ', np.array(epoch_train_kl_losses).mean())
            print('epoch train_loss: ', np.array(epoch_train_losses).mean())
            print('epoch  evaluation_rec_loss: ', np.array(epoch_ev_rec_loss).mean())
            print('epoch  evaluation_kl_loss: ', np.array(epoch_ev_kl_loss).mean())
            print('epoch  evaluation_loss: ', np.array(epoch_ev_loss).mean())
            print('_______________________________________________')
            #save_model_file = args.dirpath + "archive_003/cvae"
            #conv_vae.saver.save(conv_vae.sess, save_model_file, global_step=epoch)
            save_model_file = args.dirpath + "archive_003/cvae_weights_" + str(epoch)
            conv_vae.save_weights(save_model_file)
    #saves last model
    #save_model_file = args.dirpath + "archive_003/cvae_last"
    #conv_vae.saver.save(conv_vae.sess, save_model_file)
    save_model_file = args.dirpath + "archive_003/cvae_weights_last"
    conv_vae.save_weights(save_model_file)
    #last log
    print('overall train_loss: ', np.array(train_loss).mean())
    print('overall evaluation_loss: ', np.array(ev_loss).mean())
    """Test"""
    img_orig = dataset['next_obs'][10]
    z, mu, log_sigma = conv_vae.encode(img_orig.copy())
    img_restored = conv_vae.decode(z)
    store_image_array_at(img_orig, args.dirpath, 'image_original_after_training')
    store_image_array_at(img_restored, args.dirpath, 'image_restored_after_training')

def transform_to_rbgarry(flattened_array):
    img_array = flattened_array.reshape(3, 84, 84).transpose((1, 2, 0))
    return img_array

def make_batches(data, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        data: np array to data to be splitted in batches
        batch_size: Integer, batch size.

    # Returns
        A array with batches.
    """
    size = len(data)
    num_batches = (size + batch_size - 1) // batch_size  # round up
    permuted_indices = np.random.permutation(size)#indexes used since wholedata is to
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(size, (i + 1) * batch_size)
        indices_to_use = permuted_indices[batch_start:batch_end]
        yield np.take(data, indices_to_use, axis=0)

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

def play_with_trained(args):
    # LOADING THE DATA
    dataset = np.load(filename, allow_pickle=True).item()
    N = len(dataset['obs'])
    # shaping as image 84x84x3
    dataset['obs'] = np.array([transform_to_rbgarry(arr) for arr in dataset['obs']]).astype(np.float32)
    dataset['next_obs'] = np.array([transform_to_rbgarry(arr) for arr in dataset['next_obs']]).astype(np.float32)
    # Normalizing the images to the range of [0., 1.]
    dataset['obs'] /= 255.
    dataset['next_obs'] /= 255.

    restore_path = None#args.dirpath + "archive_003/cvae_weights_last"
    #conv_vae = CVAE(lr=0.001, input_channels=3, representation_size=16, imsize=84, restore_path=restore_path)
    conv_vae = CVAE_Keras(lr=0.001, input_channels=3, representation_size=16, imsize=84, restore_path=restore_path)
    img_orig = dataset['next_obs'][38000]
    z, mu, log_sigma = conv_vae.encode(img_orig)
    print('this is z:', z)
    img_restored = conv_vae.decode(z)
    random_z = np.random.uniform(low=-1.5, high=2, size=16)
    img_random = conv_vae.decode(random_z)
    store_image_array_at(img_orig, args.dirpath, 'image_original')
    store_image_array_at(img_restored, args.dirpath, 'image_restored')
    store_image_array_at(img_random, args.dirpath, 'image_random')

def prove_trained_paper(args):
    # LOADING THE DATA
    dataset = np.load(filename, allow_pickle=True).item()

    vae_file = '/home/erick/log_leap/pnr/05-26-train-vae-local/05-26-train-vae-local_2020_05_26_22_58_54_id000--s57026/vae_itr_1000.pkl'
    vae = pickle.load(open(vae_file, "rb"))
    assert isinstance(vae, ConvVAE2)

    img_orig = dataset['next_obs'][10]
    img_orig_reshaped = transform_to_rbgarry(img_orig.copy())
    store_image_array_at(img_orig_reshaped, args.dirpath, 'image_original_paper')
    torch_input = ptu.np_to_var(img_orig)
    mu, logbar = vae.encode(torch_input)
    z = vae.reparameterize(mu, logbar)
    print('this is z:', z)
    img_restored = vae.decode(z)
    img_restored_reshaped = transform_to_rbgarry(ptu.get_numpy(img_restored))
    store_image_array_at(img_restored_reshaped, args.dirpath, 'image_restored_paper')

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirpath', help='path to to directory save/load the model', type=str)
    args = parser.parse_args()
    #train(args)
    #d2, d11, d1, z = prove_store(args)
    #prove_load(d2, d11, d1, z, args)
    #play_with_trained(args)
    #prove_trained_paper(args)

