import numpy as np
import tensorflow as tf
from ownimplementations.tf_utils import get_vars



class CVAE:
    def __init__(self, lr,input_channels, num_filters, action_size, representation_size=16, imsize=84):
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.lr = lr
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
                    #todo USE BATCH NORM
                    x = tf.layers.conv2d(X, filters=16, kernel_size=5, strides=3, padding='same', activation=activation)
                    x = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=3, padding='same', activation=activation)
                    x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=3, padding='same', activation=activation)
                    x = tf.layers.flatten(x)

                    # Local latent variables
                    mu = tf.layers.dense(x, units=self.representation_size, name='mean')
                    log_sigma = tf.nn.softplus(tf.layers.dense(x, units=self.representation_size),   name='std_dev')
                    # softplus to force >0

                    # Reparametrization trick
                    epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.representation_size]), name='epsilon')
                    z = mu + tf.multiply(epsilon, log_sigma)

                    return z, mu, log_sigma

            def decoder(z, scope_name):
                activation = tf.nn.relu
                with tf.variable_scope(scope_name):
                    x = tf.layers.dense(z, units=self.representation_size, activation=activation)
                    x = tf.layers.dense(x, units=self.representation_size, activation=activation)
                    recovered_size = int(np.sqrt(self.representation_size))
                    x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])

                    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=3, padding='same',
                                                   activation=activation)
                    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=3, padding='same',
                                                   activation=activation)
                    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=6, strides=3, padding='same',
                                                   activation=activation)

                    x = tf.contrib.layers.flatten(x)
                    x = tf.layers.dense(x, units=self.imsize * self.imsize, activation=None)

                    x = tf.layers.dense(x, units=self.imsize * self.imsize, activation=tf.nn.sigmoid)
                    img = tf.reshape(x, shape=[-1, self.imsize, self.imsize, self.input_channels])

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
                self.z, self.mu, self.log_sigma = encoder(self.img_input_ph, "encoder")
                self.img = decoder(self.z, 'decoder')
            with tf.variable_scope('dec'):
                self.img_dec = decoder(self.z_dec_ph, "decoder")


        def create_operators():
            flat_input = tf.reshape(self.img_input_ph, (-1,))
            flat_output = tf.reshape(self.img, (-1,))
            self.reconstruction_loss = tf.reduce_sum(tf.square(flat_output - flat_input))
            self.kl_loss = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.log_sigma) - tf.log(tf.square(self.log_sigma)) - 1, 1)
            self.loss = tf.reduce_mean(self.reconstruction_loss + self.kl_loss)

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, var_list=get_vars('vae'))

            self.dec_z_update_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('vae/decoder'), get_vars('dec/decoder'))
            ])

            self.saver = tf.train.Saver(max_to_keep=100)
            self.init_op = tf.global_variables_initializer()
            self.dec_z_init_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('vae/decoder'), get_vars('dec/decoder'))
            ])

            ###


        self.graph = tf.Graph()
        with self.graph.as_default():
            create_session()
            create_inputs()
            create_network()
            create_operators()
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.dec_z_init_op)

    '''def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.steps_counter < self.args.warmup):
            return np.random.uniform(-1, 1, size=self.args.acts_dims)
        if self.args.goal_based: obs = goal_based_process(obs)

        # eps-greedy exploration
        if explore and np.random.uniform() <= self.args.eps_act:
            return np.random.uniform(-1, 1, size=self.args.acts_dims)

        feed_dict = {
            self.raw_obs_ph: [obs]
        }
        action, info = self.sess.run([self.pi, self.step_info], feed_dict)
        action = action[0]

        # uncorrelated gaussian explorarion
        if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
        action = np.clip(action, -1, 1)

        if test_info: return action, info
        return action

    def step_batch(self, obs):
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions'''


    def train(self, batch):
        loss, _, = self.sess.run([self.loss, self.train_op], {self.img_input_ph: batch})
        self.sess.run(self.dec_z_update_op)
        return loss

    def evaluate(self, batch):
        loss = self.sess.run(self.loss, {self.img_input_ph: batch})
        return loss


    def init_weights(self):
        pass

    def encode(self, im):
        z, mu, log_sigma = self.sess.run(self.z, self.mu, self.log_sigma, {self.img_input_ph: [im]})
        return z, mu, log_sigma

    def decode(self, z):
        img_dec = self.ses.run(self.img_dec, {self.z_dec_ph: [z]})
        return img_dec



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
