import tensorflow as tf
import numpy as np

class VGG16:
    
    def __init__(self,weights, sess):
        self.initialInput()
        self.convolutionLayers()
        self.fullyConnectedLayers()
        self.output_prediction = tf.nn.softmax(self.fully_connected3)
        self.loadWeights(weights, sess)
    
    def initialInput(self):
        #self.input_image = tf.placeholder(tf.string, shape=[])
        #self.decode_jpeg = tf.image.decode_jpeg(self.input_image, channels=3, ratio = 1, fancy_upscaling=True, try_recover_truncated=False)
        #self.cast = tf.cast(self.decode_jpeg, tf.float32)
        #self.multi = tf.expand_dims(self.cast, 0)
        #self.sized = tf.image.resize_bilinear(self.multi, [224, 224], align_corners=False)
        #self.input_image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self.input_image = tf.placeholder(tf.float32, [None, 224, 224, 3])
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        self.image = self.input_image-mean
        
    def convolutionLayers(self):
        self.component = []
        #conv1.1
        #kernel patch size,,channels, output
        weights = tf.truncated_normal([3, 3, 3, 64], stddev = 0.1)
        weights = tf.Variable(weights)

        bias = tf.constant(0.1, shape=[64])
        bias = tf.Variable(bias)

        conv = tf.nn.conv2d(self.image, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv11 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv1.2
        #kernel patch size,,channels, output
        weights = tf.truncated_normal([3, 3, 64, 64], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[64])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv11, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv12 = tf.nn.relu(output)
        self.component += [weights, bias]

        #maxpool
        self.pool1 = tf.nn.max_pool(self.conv12, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding='SAME', name='pool1')

        #conv2.1
        #kernel patch size,,channels, output
        weights = tf.truncated_normal([3, 3, 64, 128], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[128])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.pool1, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv21 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv2.2
        #kernel patch size,,channels, output
        weights = tf.truncated_normal([3, 3, 128, 128], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[128])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv21, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv22 = tf.nn.relu(output)
        self.component += [weights, bias]

        #maxpool
        self.pool2 = tf.nn.max_pool(self.conv22, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding='SAME', name='pool2')

        #conv3.1
        weights = tf.truncated_normal([3, 3, 128, 256], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[256])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.pool2, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv31 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv3.2
        weights = tf.truncated_normal([3, 3, 256, 256], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[256])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv31, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv32 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv3.3
        weights = tf.truncated_normal([3, 3, 256, 256], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[256])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv32, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv33 = tf.nn.relu(output)
        self.component += [weights, bias]

        #maxpool
        self.pool3 = tf.nn.max_pool(self.conv33, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding='SAME', name='pool3')

        #conv4.1
        weights = tf.truncated_normal([3, 3, 256, 512], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[512])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.pool3, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv41 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv4.2
        weights = tf.truncated_normal([3, 3, 512, 512], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[512])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv41, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv42 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv4.3
        weights = tf.truncated_normal([3, 3, 512, 512], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[512])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv42, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv43 = tf.nn.relu(output)
        self.component += [weights, bias]

        #maxpool
        self.pool4 = tf.nn.max_pool(self.conv43, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding='SAME', name='pool4')

        #conv5.1
        weights = tf.truncated_normal([3, 3, 512, 512], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[512])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.pool4, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv51 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv5.2
        weights = tf.truncated_normal([3, 3, 512, 512], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[512])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv51, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv52 = tf.nn.relu(output)
        self.component += [weights, bias]

        #conv5.3
        weights = tf.truncated_normal([3, 3, 512, 512], stddev = 0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[512])
        bias = tf.Variable(bias)
        conv = tf.nn.conv2d(self.conv52, weights, [1, 1, 1, 1], padding='SAME')
        output = tf.nn.bias_add(conv, bias)
        self.conv53 = tf.nn.relu(output)
        self.component += [weights, bias]

        #maxpool
        self.pool5 = tf.nn.max_pool(self.conv53, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding='SAME', name='pool5')

    def fullyConnectedLayers(self):
        #FC4k
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        weights = tf.truncated_normal([shape, 4096],stddev=0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[4096])
        bias = tf.Variable(bias)
        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        output = tf.nn.bias_add(tf.matmul(pool5_flat, weights), bias)
        self.fully_connected1 = tf.nn.relu(output)
        self.component += [weights, bias]

        #FC4k
        weights = tf.truncated_normal([4096, 4096],stddev=0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[4096])
        bias = tf.Variable(bias)
        output = tf.nn.bias_add(tf.matmul(self.fully_connected1, weights), bias)
        self.fully_connected2 = tf.nn.relu(output)
        self.component += [weights, bias]

        #FC1k
        weights = tf.truncated_normal([4096, 1000],stddev=0.1)
        weights = tf.Variable(weights)
        bias = tf.constant(0.1, shape=[1000])
        bias = tf.Variable(bias)
        output = tf.nn.bias_add(tf.matmul(self.fully_connected2, weights), bias)
        self.fully_connected3 = tf.nn.relu(output)
        self.component += [weights, bias]
     
    def loadWeights(self, weights, sess):
        weight = np.load(weights)
        keys = sorted(weight.keys())
        for i, k in enumerate(keys):
            #print i, k, np.shape(weights[k])
            sess.run(self.component[i].assign(weight[k]))
            
    def testModel(self, image, sess):
        prob = sess.run(self.output, feed_dict={self.input_image: [image]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])