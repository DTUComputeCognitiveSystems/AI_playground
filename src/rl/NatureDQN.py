import tensorflow as tf
from src.rl.BaseNet import BaseNet


class NatureDQN(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            # placeholders
            self.state = tf.placeholder("float",
                                        [None, config['in_width'], config['in_height'], config['state_frames']],
                                        name='state')
            self.nstate = tf.placeholder("float",
                                         [None, config['in_width'], config['in_height'], config['state_frames']],
                                         name='nstate')
            self.rewards = tf.placeholder("float", [None], name='rewards')
            self.actions = tf.placeholder("float", [None, config['num_actions']], name='actions')  # one-hot
            self.terminals = tf.placeholder("float", [None], name='terminals')

            outputs = [self.state]
            outputs_target = [self.nstate]
            self.assign_ops = []
            self.conv = []

            # region make conv layers
            for n in range(config['conv_layers']):
                with tf.variable_scope('conv' + str(n)) as scope:
                    shape = [config['filter_sizes'][n],
                             config['filter_sizes'][n],
                             config['state_frames'] if n == 0 else config['conv_units'][n-1],
                             config['conv_units'][n]]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['conv_units'][n])
                    conv = self.conv2d(outputs[-1], W, config['strides'][n])
                    conv = tf.nn.bias_add(conv, b)
                    conv = tf.nn.relu(conv, name=scope.name)

                    # EXPERIMENT: zero out feature maps
                    if 'drop_experiment' in config:
                        nlayer = config['drop_nlayer']
                        nmap = config['drop_nmaps']
                        if n == nlayer:
                            splits = tf.split(3, config['conv_units'][n], conv)
                            for m in nmap:
                                splits[m] = tf.zeros_like(splits[m])
                            conv = tf.concat(3, splits)
                    # END TODO


                    outputs.append(conv)
                    self.conv.append(conv)

                    # target network and assign ops
                    W_target = tf.Variable(W.initialized_value(), trainable=False)
                    b_target = tf.Variable(b.initialized_value(), trainable=False)
                    conv_target = self.conv2d(outputs_target[-1], W_target, config['strides'][n])
                    conv_target = tf.nn.bias_add(conv_target, b_target)
                    conv_target = tf.nn.relu(conv_target, name=scope.name + '_target')
                    outputs_target.append(conv_target)
                    W_op = W_target.assign(W)
                    b_op = b_target.assign(b)
                    self.assign_ops.append(W_op)
                    self.assign_ops.append(b_op)

                    # tensorboard
                    #if n == 0:  # just first layer
                    #    for (i, img) in enumerate(tf.split(3, config['conv_units'][n], W)):
                    #        img = tf.transpose(img, [2,0,1,3])
                    #        tf.image_summary(scope.name + "/W" + str(i), img, max_images=4)
            # endregion make conv layers

            # region deconv visualization of layer 1
            with tf.variable_scope('conv0', reuse=True):
                shape = [config['filter_sizes'][0],
                         config['filter_sizes'][0],
                         config['state_frames'],
                         config['conv_units'][0]]
                W = self.make_weight(shape)
                b = self.make_bias(config['conv_units'][0])

                # process visualization layer
                conv_slice = tf.slice(self.conv[0], [0, 0, 0, 0], [1, -1, -1, -1])  # slice batch
                conv_slice_split = tf.split(conv_slice, conv_slice.get_shape()[-1], 3)
                conv_slice_units = 1
                conv_slice_shape = conv_slice.get_shape().as_list()
                for d in conv_slice_shape[1:-1]:
                    conv_slice_units *= d
                conv_slice_flat = tf.reshape(conv_slice, [conv_slice_units, conv_slice_shape[-1]])
                index_max = tf.argmax(conv_slice_flat, 0)
                one_hot_index = tf.one_hot(index_max, conv_slice_units, 1.0, 0.0)
                conv_slice_mask = tf.reshape(one_hot_index, conv_slice.get_shape())
                masked_conv = tf.multiply(conv_slice_mask, conv_slice)
                # we need a batch of size #feature maps
                masked_conv_split = tf.split(masked_conv, conv_slice_shape[-1], 3)
                batch_maps = []
                for i in range(conv_slice_shape[-1]):
                    zero_splits = tf.split(tf.zeros_like(self.conv[0]), config['conv_units'][0], 3)
                    zero_splits[i] = conv_slice_split[i]#masked_conv_split[i]
                    batch_maps.append(tf.concat(zero_splits, 3))
                #zero_splits = tf.split(3, config['conv_units'][nlayer], tf.zeros_like(self.conv[nlayer]))
                #zero_splits[nmap] = conv_slice#tf.mul(conv_slice, conv_slice_mask)
                #zero_maps = tf.concat(3, zero_splits)

                # deconv each layer
                self.deconv = []
                for m in batch_maps:
                    output_shape = tf.stack([tf.shape(self.state)[0],
                                            tf.shape(m)[1] * config['strides'][0],
                                            tf.shape(m)[2] * config['strides'][0],
                                            config['state_frames']])
                    strides = [1, config['strides'][0], config['strides'][0], 1]
                    deconv = tf.nn.conv2d_transpose(m, W, output_shape, strides, "SAME")
                    deconv = tf.nn.relu(deconv)
                    self.deconv.append(deconv)
            # endregion deconv visualization

            # region make fc layers
            conv_neurons = 1
            for d in outputs[-1].get_shape()[1:].as_list():
                conv_neurons *= d
            self.reshape = tf.reshape(outputs[-1], [-1, conv_neurons], name='reshape')
            outputs.append(self.reshape)

            self.reshape_target = tf.reshape(outputs_target[-1], [-1, conv_neurons], name='reshape_target')
            outputs_target.append(self.reshape_target)

            for n in range(config['fc_layers']):
                with tf.variable_scope('fc' + str(n)) as scope:
                    shape = [conv_neurons if n == 0 else config['fc_units'][n-1],
                             config['fc_units'][n]]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['fc_units'][n])
                    fc = tf.nn.relu_layer(outputs[-1], W, b, name=scope.name)
                    outputs.append(fc)

                    # target network and assign ops
                    W_target = tf.Variable(W.initialized_value(), trainable=False)
                    b_target = tf.Variable(b.initialized_value(), trainable=False)
                    fc_target = tf.nn.relu_layer(outputs_target[-1], W_target, b_target, name=scope.name + '_target')
                    outputs_target.append(fc_target)
                    W_op = W_target.assign(W)
                    b_op = b_target.assign(b)
                    self.assign_ops.append(W_op)
                    self.assign_ops.append(b_op)
            # endregion make fc layers

            # region output layer
            with tf.variable_scope('output') as scope:
                shape = [config['fc_units'][-1],
                         config['num_actions']]
                W = self.make_weight(shape)
                b = self.make_bias(config['num_actions'])
                self.Q = tf.nn.bias_add(tf.matmul(outputs[-1], W), b, name=scope.name + '_Q')
                outputs.append(self.Q)
                self.argmax_Q = tf.argmax(self.Q, axis=1, name=scope.name + '_argmax_Q')
                outputs.append(self.argmax_Q)

                # target network and assign ops
                W_target = tf.Variable(W.initialized_value(), trainable=False)
                b_target = tf.Variable(b.initialized_value(), trainable=False)
                self.Q_target = tf.nn.bias_add(tf.matmul(outputs_target[-1],
                                                         W_target), b_target, name=scope.name + '_Q_target')
                outputs_target.append(self.Q_target)
                self.max_Q_target = tf.reduce_max(self.Q_target, 1, name=scope.name + '_max_Q_target')
                W_op = W_target.assign(W)
                b_op = b_target.assign(b)
                self.assign_ops.append(W_op)
                self.assign_ops.append(b_op)
            # endregion output layer

            # region cost
            self.discount = tf.constant(config['discount'])
            self.y = tf.add(self.rewards, tf.multiply(self.discount, tf.multiply(tf.subtract(1.0, self.terminals), self.max_Q_target)))
            self.Q_action = tf.reduce_sum(tf.multiply(self.Q, self.actions), reduction_indices=1)  # TODO: see Tensorflow#206

            # td error clipping
            self.clip_delta = tf.constant(config['clip_delta'])
            self.diff = tf.subtract(self.y, self.Q_action)
            self.quadratic_part = tf.minimum(tf.abs(self.diff), self.clip_delta)
            self.linear_part = tf.subtract(tf.abs(self.diff), self.quadratic_part)
            self.clipped_diff = tf.add(0.5 * tf.square(self.quadratic_part),
                                       tf.multiply(self.clip_delta, self.linear_part))
            self.cost = tf.reduce_sum(self.clipped_diff, reduction_indices=0)
            # endregion cost

            self.optimize_op = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                         config['momentum'], config['opt_eps']).minimize(self.cost)

        super(NatureDQN, self).__init__(config)

    def sync_target(self):
        self.sess.run(self.assign_ops)

    def train(self, s, a, r, ns, t):
        feed_dict = {self.state: s/255.0, self.actions: a, self.rewards: r, self.nstate: ns/255.0, self.terminals: t}

        cost, _ = self.sess.run([self.cost, self.optimize_op], feed_dict)

        return cost

    def predict(self, s):
        feed_dict = {self.state: s/255.0}

        argmax_Q = self.sess.run(self.argmax_Q, feed_dict)[0]

        return argmax_Q

    def visualize(self, s):
        feed_dict = {self.state: s/255.0}

        return self.sess.run(self.deconv, feed_dict)
