import tensorflow as tf


class BaseNet(object):
    def __init__(self, config):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False,
                                                     inter_op_parallelism_threads=4,
                                                     intra_op_parallelism_threads=4))  # TF uses all cores by default...
        self.saver = tf.train.Saver()

        self.tensorboard = config['tensorboard']
        if self.tensorboard:
            self.merged = tf.merge_all_summaries()
            self.writer = tf.train.SummaryWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def save(self, name):
        self.saver.save(self.sess, "save/model_" + name + ".ckpt")

    def load(self, name):
        self.saver.restore(self.sess, "save/model_" + name + ".ckpt")
        if self.tensorboard:  # generate a tensorboard summary from the loaded model
            summaries = self.sess.run(self.merged)
            self.writer.add_summary(summaries)
            self.writer.flush()

    @staticmethod
    def make_weight(shape):
        return tf.get_variable('weight', shape,
                               initializer=tf.initializers.variance_scaling(scale=1.43, distribution="uniform"))  # 1.43 for relu

    @staticmethod
    def make_bias(shape):
        return tf.get_variable('bias', shape,
                               initializer=tf.constant_initializer(0.001))

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def conv2d_transpose(x, W, shape, stride):
        return tf.nn.conv2d_transpose(x, W, shape, strides=[1, stride, stride, 1], padding="SAME")
