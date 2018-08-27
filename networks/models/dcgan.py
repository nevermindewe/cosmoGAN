import sys

import tensorflow as tf
from .ops import linear, conv2d, conv2d_transpose, lrelu, average_gradients, get_available_gpus


class DCGAN(object):
    def __init__(self, output_size=64, batch_size=64,
                 nd_layers=4, ng_layers=4, df_dim=128, gf_dim=128,
                 c_dim=1, z_dim=100, flip_labels=0.01, data_format="NHWC",
                 gen_prior=tf.random_normal, transpose_b=False,
                 num_gpus=-1, learning_rate=.0002, beta1=0.5):

        self.output_size = output_size
        self.batch_size = batch_size
        self.nd_layers = nd_layers # Number of hidden layers in the discriminator?
        self.ng_layers = ng_layers # Number of hidden layers in the generator?
        self.df_dim = df_dim # discriminator f dimensions?
        self.gf_dim = gf_dim # generator f dimensions
        self.c_dim = c_dim # number of channels
        self.z_dim = z_dim # 
        self.flip_labels = flip_labels
        self.data_format = data_format
        self.gen_prior = gen_prior
        self.transpose_b = transpose_b  # transpose weight matrix in linear layers for (possible) better performance when running on HSW/KNL
        self.stride = 2  # this is fixed for this architecture
        
        self.compute_devices = get_available_gpus()
        # Figure out if we are using gpus, how many, which ones
        # If num_gpus < 0, use all GPUs we were given
        if num_gpus > 0:
            # If the user doesn't want to use all the gpus, take the first N
            self.compute_devices = self.compute_devices[:num_gpus]
        elif num_gpus == 0:
            self.compute_devices = ['/cpu:0']

        self._check_architecture_consistency()

        self.batchnorm_kwargs = {'epsilon': 1e-5, 'decay': 0.9,
                                 'updates_collections': None, 'scale': True,
                                 'fused': True, 'data_format': self.data_format}
        self.make_optimizers(learning_rate, beta1)

    @property
    def compute_batch_size(self):
        """Return the batch size that each GPU (or other compute device) should have"""
        return self.batch_size / len(self.compute_devices)
        
    def training_graph(self):
        """Make the training graph such that it divides the work among 1 or more GPUS"""

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')
            
        self.z = self.gen_prior(shape=[self.batch_size, self.z_dim])

        # Take the batch of images and split the generator and discriminator work among
        #   all the computing devices given.
        # My hope is that by parallelizing this inner graph, we'll get some
        #   nice speedup without having to refactor a lot.
        real_image_splits = tf.split(self.images, num_or_size_splits=len(self.compute_devices), axis=0)
        # Split the random-noise images into chunks, one for each compute device
        gen_image_splits = tf.split(self.z, num_or_size_splits=len(self.compute_devices), axis=0)

        tower_d_gradients = []
        tower_g_gradients = []
        with tf.variable_scope(tf.get_variable_scope()):
            for idx, (images, z_images) in enumerate(zip(real_image_splits, gen_image_splits)):
                compute_device = self.compute_devices[idx]
                with tf.device(compute_device):
                    with tf.variable_scope("tower_%d" % idx) as scope:
                        with tf.variable_scope("discriminator"):
                            d_prob_real, d_logits_real = self.discriminator(images, is_training=True)

                        with tf.variable_scope("generator"):
                            g_images = self.generator(z_images, is_training=True)

                        with tf.variable_scope("discriminator") as d_scope:
                            d_scope.reuse_variables()
                            d_prob_fake, d_logits_fake = self.discriminator(g_images, is_training=True)

                        with tf.name_scope("losses"):
                            with tf.name_scope("d"):
                                d_label_real, d_label_fake = self._labels()
                                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=d_label_real, name="real"))
                                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=d_label_fake, name="fake"))
                                d_loss = d_loss_real + d_loss_fake

                            with tf.name_scope("g"):
                                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

                        # Each tower should reuse the same variables
                        #   across towers so that when we apply the 
                        #   averaged gradients, we only have to update each variable
                        #   once, instead of each variable per tower.
                        tf.get_variable_scope().reuse_variables()
                        
                        t_vars = tf.trainable_variables()
                        d_vars = [var for var in t_vars if 'discriminator/' in var.name]
                        g_vars = [var for var in t_vars if 'generator/' in var.name]

                        d_gradients, g_gradients = self.compute_gradients(d_loss, g_loss, d_vars, g_vars)
                        tower_d_gradients.append(d_gradients)
                        tower_g_gradients.append(g_gradients)

        # Do these variables need to be separated by variable name so that
        #   the average_gradients function will correctly aggregate them?
        # I think so.
        g_grads_vars = average_gradients(tower_g_gradients)
        d_grads_vars = average_gradients(tower_d_gradients)

        # FIXME: These variables should be removed once I figure out how to make them
        #   available through the summary.
        # FIXME: These variables only hold the last result from the data-parallel GPU
        #   operations. 
        self.d_loss_fake = d_loss_fake
        self.d_loss_real = d_loss_real
        self.g_loss = g_loss
        
        self.d_summary = tf.summary.merge([tf.summary.histogram("prob/real", d_prob_real),
                                           tf.summary.histogram("prob/fake", d_prob_fake),
                                           tf.summary.scalar("loss/real", d_loss_real),
                                           tf.summary.scalar("loss/fake", d_loss_fake),
                                           tf.summary.scalar("loss/d", d_loss)])

        g_sum = [tf.summary.scalar("loss/g", g_loss)]
        if self.data_format == "NHWC":  # tf.summary.image is not implemented for NCHW
            g_sum.append(tf.summary.image("G", g_images, max_outputs=4))
        self.g_summary = tf.summary.merge(g_sum)

        with tf.variable_scope("counters"):
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)

        return g_grads_vars, d_grads_vars
        
    def inference_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        with tf.variable_scope("discriminator"):
            self.D, _ = self.discriminator(self.images, is_training=False)

        with tf.variable_scope("generator"):
            self.G = self.generator(self.z, is_training=False)

        with tf.variable_scope("counters"):
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)

    def make_optimizers(self, learning_rate, beta1):
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        
    def compute_gradients(self, d_loss, g_loss, d_vars, g_vars):
        d_optim = self.d_optim.compute_gradients(d_loss, var_list=d_vars)
        g_optim = self.g_optim.compute_gradients(g_loss, var_list=g_vars)

        return d_optim, g_optim

    def apply_gradients(self, g_grads_vars, d_grads_vars):
        return tf.group(self.g_optim.apply_gradients(g_grads_vars),
                        self.d_optim.apply_gradients(d_grads_vars, global_step=self.global_step))

    def generator(self, z, is_training):
        map_size = self.output_size / int(2**self.ng_layers)
        num_channels = self.gf_dim * int(2**(self.ng_layers - 1))

        z_ = linear(z, num_channels*map_size*map_size, 'h0_lin', transpose_b=self.transpose_b)
        h0 = tf.reshape(z_, self._tensor_data_format(-1, map_size, map_size, num_channels))
        bn0 = tf.contrib.layers.batch_norm(h0, is_training=is_training, scope='bn0', **self.batchnorm_kwargs)
        h0 = tf.nn.relu(bn0)

        chain = h0

        for h in range(1, self.ng_layers):
            map_size *= self.stride
            num_channels /= 2
            chain = conv2d_transpose(chain,
                                     self._tensor_data_format(self.compute_batch_size, map_size, map_size, num_channels),
                                     stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T' % h)
            chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i' % h, **self.batchnorm_kwargs)
            chain = tf.nn.relu(chain)

        map_size *= self.stride
        hn = conv2d_transpose(chain,
                              self._tensor_data_format(self.compute_batch_size, map_size, map_size, self.c_dim),
                              stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T' % self.ng_layers)

        return tf.nn.tanh(hn)

    def discriminator(self, image, is_training):
        """This discriminator works on a single image at a time"""

        chain = lrelu(conv2d(image, self.df_dim, self.data_format, name='h0_conv'))

        for h in range(1, self.nd_layers):
            chain = conv2d(chain, self.df_dim*(2**h), self.data_format, name='h%i_conv' % h)
            chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i' % h, **self.batchnorm_kwargs)
            chain = lrelu(chain)

        hn = linear(tf.reshape(chain, [self.batch_size, -1]),
                    1,
                    'h%i_lin' % self.nd_layers,
                    transpose_b=self.transpose_b)

        return tf.nn.sigmoid(hn), hn

    def _labels(self):
        with tf.name_scope("labels"):
            ones = tf.ones([self.batch_size, 1])
            zeros = tf.zeros([self.batch_size, 1])
            flip_labels = tf.constant(self.flip_labels)

            if self.flip_labels > 0:
                prob = tf.random_uniform([], 0, 1)

                d_label_real = tf.cond(tf.less(prob, flip_labels), lambda: zeros, lambda: ones)
                d_label_fake = tf.cond(tf.less(prob, flip_labels), lambda: ones, lambda: zeros)
            else:
                d_label_real = ones
                d_label_fake = zeros

        return d_label_real, d_label_fake

    def _tensor_data_format(self, N, H, W, C):
        if self.data_format == "NHWC":
            return [int(N), int(H), int(W), int(C)]
        else:
            return [int(N), int(C), int(H), int(W)]

    def _check_architecture_consistency(self):

        if self.output_size/2**self.nd_layers < 1:
            print("Error: Number of discriminator conv. layers are larger than the output_size for this architecture")
            exit(0)

        if self.output_size/2**self.ng_layers < 1:
            print("Error: Number of generator conv_transpose layers are larger than the output_size for this architecture")
            exit(0)

