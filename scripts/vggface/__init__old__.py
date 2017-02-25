import h5py
import tensorflow as tf
import numpy as np
import cv2
import os



class VGGFace(object):

    def __init__(self,):
        self.params = None
        self.batch_size = 1
        self.vars = []
        self.layers = []
        self.names = [line.strip() for line in file(os.path.join(os.path.dirname(os.path.realpath("__file__")), 'vggface/names.txt'))]
        # (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','1',3,3,3,64))
        # (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','3',3,3,64,64))
        # (5): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','6',3,3,64,128))
        # (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','8',3,3,128,128))
        # (10): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','11',3,3,128,256))
        # (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','13',3,3,256,256))
        # (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','15',3,3,256,256))
        # (17): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','18',3,3,256,512))
        # (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','20',3,3,512,512))
        # (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','22',3,3,512,512))
        # (24): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','25',3,3,512,512))
        # (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','27',3,3,512,512))
        # (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','29',3,3,512,512))
        # (31): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (32): nn.View
        # (33): nn.Linear(25088 -> 4096)
        self.layers.append(('linear','33',4096,True))
        # (34): nn.ReLU
        # (35): nn.Dropout(0.500000)
        # (36): nn.Linear(4096 -> 4096)
        self.layers.append(('linear','36',4096,True))
        #self.layers.append(('l2','37',4096,True))
        
        
        
        # (37): nn.ReLU
        # (38): nn.Dropout(0.500000)
        # (39): nn.Linear(4096 -> 2622)
        # self.layers.append(('linear','39',2622,False))
        # (40): nn.SoftMax
        # self.layers.append(('softmax'))

    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var,layer):
        self.vars.append((name, var,layer))

    def get_output(self):
        return self.vars[-1][1]

    def make_var(self, name, shape):
        return tf.get_variable(name, shape)

    def setup(self):
        for layer in self.layers:
            name = self.get_unique_name_(layer[0])
            if layer[0] == 'conv':
                with tf.variable_scope(name) as scope:
                    h, w, c_i, c_o = layer[2],layer[3],layer[4],layer[5]
                    kernel = self.make_var('weights', shape=[h, w, c_i, c_o])
                    conv = tf.nn.conv2d(self.get_output(), kernel, [1]*4, padding='SAME')
                    biases = self.make_var('biases', [c_o])
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    relu = tf.nn.relu(bias, name=scope.name)
                    self.add_(name, relu,layer)
            elif layer[0] == 'pool':
                size,size,stride,stride = layer[1],layer[2],layer[3],layer[4]
                pool = tf.nn.max_pool(self.get_output(),
                                      ksize=[1, size, size, 1],
                                      strides=[1, stride, stride, 1],
                                      padding='SAME',
                                      name=name)
                self.add_(name, pool,layer)
            elif layer[0] == 'linear':
                num_out = layer[2]
                relu = layer[3]
                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    weights = self.make_var('weights', shape=[dim, num_out])
                    biases = self.make_var('biases', [num_out])
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(feed_in, weights, biases, name=scope.name)
                    self.add_(name, fc,layer)
            elif layer[0] == 'softmax':
                self.add_(name, tf.nn.softmax(self.get_output()),layer)
            elif layer[0] == 'l2':
                self.add_(name,tf.nn.l2_normalize(self.get_output(),0,),layer)

    def load(self, ses, input_img, path = os.path.join(os.path.dirname(os.path.realpath("__file__")), 'vggface/network.h5')):
        self.params = h5py.File(path,'r')
        self.vars.append(('input',input_img,['input',None]))
        self.setup()
        for name,varb,layer in self.vars:
            if layer[0] == 'conv':
                with tf.variable_scope(name, reuse=True):
                    h, w, c_i, c_o = layer[2],layer[3],layer[4],layer[5]
                    filters = np.array([k.reshape(c_i,h,w) for k in self.params[layer[1]]]).transpose((2, 3, 1, 0))
                    ses.run(tf.get_variable('weights').assign(filters))
                    ses.run(tf.get_variable('biases').assign(np.array(self.params[layer[1]+'b'])))
                    # print name,filters.shape
            if layer[0] == 'linear':
                if layer[1] == '33':
                    with tf.variable_scope(name, reuse=True):
                        filters = np.array([k for k in self.params[layer[1]]])
                        prev_c_o = 512
                        # print 'initial',filters.shape
                        cur_c_i, cur_c_o = 25088,4096
                        dim = np.sqrt(cur_c_i/prev_c_o)
                        filters = filters.reshape((cur_c_o,prev_c_o, dim, dim))
                        # print 'reshaped',filters.shape
                        filters = filters.transpose((2, 3, 1, 0))
                        # print 'transposed',filters.shape
                        filters = filters.reshape((prev_c_o*dim*dim, cur_c_o))
                        # print 'reshpaed',filters.shape
                        ses.run(tf.get_variable('weights').assign(filters))
                        ses.run(tf.get_variable('biases').assign(np.array(self.params[layer[1]+'b'])))
                        # print name,filters.shape
                else:
                    with tf.variable_scope(name, reuse=True):
                        filters = np.array([k for k in self.params[layer[1]]])
                        filters = filters.transpose((1, 0))
                        ses.run(tf.get_variable('weights').assign(filters))
                        ses.run(tf.get_variable('biases').assign(np.array(self.params[layer[1]+'b'])))
                        # print name,filters.shape

    def eval(self,*args,**kwargs):
        return  self.get_output().eval(*args,**kwargs)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img -= [129.1863,104.7624,93.5940]
    img = np.array([img,])
    return img
