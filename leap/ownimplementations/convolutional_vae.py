import numpy as np
import tensorflow as tf
from ownimplementations.tf_utils import get_vars



class CVAE:
    def __init__(self, lr,input_channels, action_size, representation_size=16, imsize=84, restore_path=None):
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.lr = lr
        self.restore_path = restore_path
        self.create_model()

    def create_model(self):


        def create_session():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        def create_inputs():
            self.img_input_ph = tf.placeholder(tf.float32, [None] + [self.imsize, self.imsize, self.input_channels],
                                               name='input_im_ph')# TODO: changed names!
            self.z_dec_ph = tf.placeholder(tf.float32, [None] + [self.representation_size], name='z_ph_for_dec')

        def create_network():
            def encoder(X, scope_name):
                activation = tf.nn.relu
                with tf.variable_scope(scope_name):
                    x = tf.layers.conv2d(X, filters=16, kernel_size=5, strides=(3,3), padding='valid',
                                         activation=activation)
                    x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=(3,3), padding='valid',
                                         activation=activation)
                    x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=(3,3), padding='valid',
                                         activation=activation)
                    x = tf.layers.flatten(x)
                    # Local latent variables
                    mu = tf.layers.dense(x, units=self.representation_size, name='mean')
                    log_sigma = tf.layers.dense(x, units=self.representation_size, name='std_dev')

                    # Reparametrization trick
                    epsilon = tf.random_normal(shape=tf.shape(log_sigma),
                                               mean=0, stddev=1, dtype=tf.float32)
                    z = mu + tf.exp(0.5 * log_sigma) * epsilon
                    return z, mu, log_sigma

            def decoder(z, scope_name):
                activation = tf.nn.relu
                with tf.variable_scope(scope_name):
                    '''x = tf.layers.dense(z, units=self.representation_size, activation=activation)
                    x = tf.layers.dense(x, units=self.representation_size, activation=activation)
                    recovered_size = int(np.sqrt(self.representation_size))
                    x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])'''
                    x = tf.layers.dense(z, units=128, activation=activation)
                    x = tf.reshape(x, [-1, 2, 2, 32])
                    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=(3,3), padding='valid',
                                                   activation=activation)
                    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=(3,3), padding='valid',
                                                   activation=activation)
                    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=6, strides=(3,3), padding='valid',
                                                   activation=activation)

                    # No activation
                    img = tf.layers.conv2d_transpose(x,filters=3, kernel_size=6, strides=(1,1), padding='same')
                    return img
                '''def linear(obs, next_obs_, actions_ph):
                    tf.layers.
                    pass
    
                latent_obs_mu, latent_obs_logvar = self.model.encode(obs)
                latent_next_obs_mu, latent_next_obs_logvar = self.model.encode(next_obs)
                action_obs_pair = torch.cat([latent_obs_mu, actions], dim=1)
                prediction = self.model.linear_constraint_fc(action_obs_pair)
                scaling = 1.0
                return torch.norm(scaling * (prediction - latent_next_obs_mu)) ** 2 / self.batch_size'''

            with tf.variable_scope('vae'):
                self.z, self.mu, self.log_sigma = encoder(self.img_input_ph, 'encoder')
                self.img = decoder(self.z, 'decoder')
            with tf.variable_scope('dec'):
                self.img_dec = decoder(self.z_dec_ph, 'decoder')


        def create_operators():
            flat_input = tf.reshape(self.img_input_ph, [-1, self.imsize*self.imsize*self.input_channels])
            flat_output = tf.reshape(self.img, [-1, self.imsize*self.imsize*self.input_channels])
            self.reconstruction_loss = tf.reduce_mean(tf.square(flat_output - flat_input), axis = -1)
            self.kl_loss = -0.5 * tf.reduce_sum(
                1 + self.log_sigma - tf.square(self.mu) - tf.exp(self.log_sigma), axis=-1)
            self.loss = tf.reduce_mean(self.reconstruction_loss + self.kl_loss)

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, var_list=get_vars('vae'))
            self.optimizer

            self.dec_z_update_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('vae/decoder'), get_vars('dec/decoder'))
            ])

            self.init_op = tf.global_variables_initializer()
            ###
        def create_saver():
            self.saver = tf.train.Saver(max_to_keep=100)

        if tf.test.gpu_device_name():
            print('using gpu')
            device_name = "/gpu:0"
        else:
            device_name = "/cpu:0"
        device_name = "/cpu:0"



        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(device_name):
                create_session()
                create_inputs()
                create_network()
                create_operators()
            create_saver()
        if self.restore_path and self.restore_path != '':
            self.saver.restore(self.sess, self.restore_path)
        else:
            self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.dec_z_update_op)

    def train(self, batch):
        z, mu, log_sigma, reconstruction_loss, kl_loss, loss, _ = self.sess.run([self.z, self.mu, self.log_sigma, self.reconstruction_loss, self.kl_loss, self.loss, self.train_op], {self.img_input_ph: batch})
        self.sess.run(self.dec_z_update_op)
        return z, mu, log_sigma, reconstruction_loss, kl_loss, loss

    def evaluate(self, batch):
        loss = self.sess.run(self.loss, {self.img_input_ph: batch})
        return loss.copy()

    def encode(self, im):
        z, mu, log_sigma = self.sess.run([self.z, self.mu, self.log_sigma], {self.img_input_ph: [im]})
        return z[0].copy(), mu[0].copy(), log_sigma[0].copy()

    def decode(self, z):
        img_dec = self.sess.run(self.img_dec, {self.z_dec_ph: [z]})
        return img_dec[0].copy()



    '''def __getstate__(self):
        d = super().__getstate__()
        # Add these explicitly in case they were modified
        d["_dist_mu"] = self.dist_mu
        d["_dist_std"] = self.dist_std
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.dist_mu = d["_dist_mu"]
        self.dist_std = d["_dist_std"]'''
