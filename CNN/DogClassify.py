from __future__ import print_function
import tensorflow as tf
import scipy.io as sp
import numpy as np

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # acc = tf.summary.scalar('accuracy',accuracy)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def show_index(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    y_preIndex = tf.argmax(y_pre,1)
    y_groundTruthIndex = tf.argmax(v_ys,1)
    
    y_preIndex = sess.run(y_preIndex, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    y_groundTruthIndex = sess.run(y_groundTruthIndex, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    
    return y_preIndex , y_groundTruthIndex




def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=0.00001)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial= tf.constant(0.00001, shape= shape)
    return tf.Variable(initial, dtype=tf.float32)

def conv2d(x, W):
    # stride  [1,x stride,y stride,1]
    # Must have strides[0] = strides[4] = 1
    return tf.nn.conv2d(x, W, strides= [1, 1, 1, 1], padding= 'VALID' )   

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1, 2 ,2 ,1], strides= [1,2,2,1], padding = 'SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize= [1, 3 ,3 ,1], strides= [1,3,3,1], padding = 'SAME')


xs = tf.placeholder(tf.float32, [None, 12288]) # 64x64x3
ys = tf.placeholder(tf.float32, [None, 22])
keep_prob = tf.placeholder(tf.float32)
x_image= tf.reshape(xs, [-1 , 64, 64, 3])  #-1 : for None in xs ; 64,64 : image size ;3 : # of channel


## conv1 layer ##
W_conv1 = weight_variable([3,3,3,32]) # patch 3x3 , input 's channel, output 's channel
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu6(conv2d(x_image, W_conv1) + b_conv1) # output size 62x62x32

## conv2 layer ##
W_conv2 = weight_variable([3,3,32,64]) # patch 3x3 , input 's channel:32, output 's channel:64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu6(conv2d(h_conv1, W_conv2) + b_conv2) # output size 60x60x64

## pool1 layer ##
h_pool1 = max_pool_2x2(h_conv2)                          # output size 30x30x64

## conv3 layer ##
W_conv3 = weight_variable([3,3,64,64]) # patch 3x3 , input 's channel, output 's channel
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu6(conv2d(h_pool1, W_conv3) + b_conv3) # output size 28x28x64

## conv4 layer ##
W_conv4 = weight_variable([3,3,64,128]) # patch 3x3 , input 's channel:64, output 's channel:128
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu6(conv2d(h_conv3, W_conv4) + b_conv4) # output size 26x26x128

## pool2 layer ##
h_pool2 = max_pool_2x2(h_conv4)                          # output size 13x13x128


## conv5 layer ##
W_conv5 = weight_variable([3,3,128,256]) # patch 3x3 , input 's channel, output 's channel
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu6(conv2d(h_pool2, W_conv5) + b_conv5) # output size 11x11x256

## conv6 layer ##
W_conv6 = weight_variable([3,3,256,256]) # patch 3x3 , input 's channel:256, output 's channel:256
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu6(conv2d(h_conv5, W_conv6) + b_conv6) # output size 9x9x256

## pool3 layer ##
h_pool3 = max_pool_3x3(h_conv6)                          # output size 3x3x256


## func1 layer ##
W_fc1 = weight_variable([3*3*256,1500])
b_fc1 = bias_variable([1500])

# [n_samples, 3, 3, 256] ->> [n_samples, 3*3*256]
h_pool3_flat = tf.reshape(h_pool3, [-1, 3*3*256]) 
h_fc1 = tf.nn.relu6(tf.matmul(h_pool3_flat,W_fc1)+ b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1500,22])
b_fc2 = bias_variable([22])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+ b_fc2)

# the error between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
    loss = tf.summary.scalar('loss',cross_entropy)


###############

# dynamic lr
learning_rate = tf.placeholder(tf.float32, shape=[])

# gradient decent method
# optimizer = tf.train.GradientDescentOptimizer(learning_rate =learning_rate) #learning rate
# train_step = optimizer.minimize(cross_entropy)



# normal Adam
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)  # AdamOptimizer is good for large data

# gradient clipping Adam
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# gvs = optimizer.compute_gradients(cross_entropy)
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# optimizer.apply_gradients(capped_gvs)
# train_step = optimizer.minimize(cross_entropy)
###############


# read data from mat.
mat_dict = {}
mat_dict.update(sp.loadmat('train_face', mdict=None, appendmat=True))
trainLabel = mat_dict['trainLabel'] # f:variable name in matlab
train = mat_dict['train'] # im:file name

mat_dict3 = {}
mat_dict3.update(sp.loadmat('test_DogFaceBody', mdict=None, appendmat=True))
testLabel = mat_dict3['testLabel'] # f:variable name in matlab
test = mat_dict3['test'] # im:file name


train = train.astype('float32')
test = test.astype('float32')
trainLabel = trainLabel.astype('float32')
testLabel = testLabel.astype('float32')


train = np.reshape(train,[train.shape[0],12288])
test = np.reshape(test,[test.shape[0],12288])
trainLabel = np.reshape(trainLabel,[train.shape[0],22])
testLabel = np.reshape(testLabel,[test.shape[0],22])


###############
batch = 32
if train.shape[0] % batch == 0:
    numToEpoch = train.shape[0] / batch
else:
    numToEpoch = train.shape[0] / batch +1
############### 

###############
#1 GPU memory saving
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#sess = tf.Session(config=config)

#2 no GPU memory saving
sess = tf.Session()

# initialized
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

# save model
saver = tf.train.Saver()
###############

lr = 1e-5

epoch = 500;

for i in range(epoch):
    for j in range(numToEpoch):
        if j == numToEpoch-1:
            sess.run(train_step, feed_dict={xs: train[j* batch : train.shape[0]- 1], ys: trainLabel[j * batch : train.shape[0]- 1], keep_prob: 0.5 ,learning_rate:lr})
        else:
            sess.run(train_step, feed_dict={xs: train[j* batch :  (j+1) * batch-1 ], ys: trainLabel[j * batch :  (j+1) * batch-1 ], keep_prob: 0.5 ,learning_rate:lr})            


save_path = saver.save(sess, "net_dog.ckpt")
print("Save to path:", save_path)

sess.close()



        
