# Dependencies for creating deep VGGish embeddings
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
pca_params = 'vggish_pca_params.npz'
model_checkpoint = 'vggish_model.ckpt'

class VGGish(object):
    """docstring for VGGish"""
    def __init__(self):
        self.graph = tf.Graph().as_default()
        self.sess = tf.Session()
        self.model = self._build_model()

    def _build_model(self):
        # Restore VGGish model trained on YouTube8M dataset
        # Retrieve PCA-embeddings of bottleneck features
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(self.sess, model_checkpoint)
        self.features_tensor = self.sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        # Prepare a postprocessor to munge the model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(pca_params)


    def predict(self, examples_batch):
        # Run inference and postprocessing.
        [embedding_batch] = self.sess.run([self.embedding_tensor],
                                     feed_dict={self.features_tensor: examples_batch})
    
        postprocessed_batch = self.pproc.postprocess(embedding_batch)
        
        return postprocessed_batch