import tensorflow as tf
import load_minst
import numpy as np
import matplotlib.pyplot as plt



#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_image = load_minst.load_mnist("training", np.arange(10) ,"/home/yuming/cogs181/hw4")[0]
train_label = load_minst.load_mnist("training", np.arange(10) ,"/home/yuming/cogs181/hw4")[1]
test_image = load_minst.load_mnist("testing", np.arange(10) ,"/home/yuming/cogs181/hw4")[0]
test_label = load_minst.load_mnist("testing", np.arange(10) ,"/home/yuming/cogs181/hw4")[1]
newtest_image = test_image.reshape(10000,784)
newtest_label = np.zeros([10000, 10])

#one hot encoding
for i in range(0,len(test_label)):
    newtest_label[i,test_label[i]] = 1
newtest_imagehalf = newtest_image[0:1000]
newtest_labelhalf = newtest_label[0:1000]

newtrain_image = train_image.reshape(60000,784)
newtrain_label = np.zeros([60000, 10])

#one hot encoding
for i in range(0,len(train_label)):
    newtrain_label[i,train_label[i,0]] = 1

batch_size = 50
def data_iterator():
    batch_idx = 0
    while True:
        index = np.arange(0, newtrain_label.shape[0])
        np.random.shuffle(index)
        shuf_features = newtrain_image[index]
        shuf_labels = newtrain_label[index]
        for batch_idx in range(0, newtrain_label.shape[0], batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch

iter_ = data_iterator()

sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#not sure the above will work, below is to CNN
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

training_times = 100
accuracy_times = int(training_times / 50)
accuracy_time = np.arange(accuracy_times)
accuracychart = np.zeros(accuracy_times)
test_accuracy_chart = np.zeros(accuracy_times)
losschart = np.zeros(training_times)

train_time = np.arange(training_times)
for i in range(training_times):
  #batch = mnist.train.next_batch(50)
  images_batch_val, labels_batch_val = next(iter_)
  if i%50 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:images_batch_val, y_: labels_batch_val, keep_prob: 1.0})
    accuracychart[int(i/50)] = train_accuracy
    print("step %d, training accuracy %g"%(i, train_accuracy))

    #test_accuracy_chart[int(i/50)] = test_accuracy
  _, loss_val = sess.run([train_step, cross_entropy],feed_dict={x: images_batch_val, y_: labels_batch_val, keep_prob: 0.5})
  losschart[i] = loss_val
  #print("loss_val: ", loss_val)
  loss_val = 0
test_accuracy = accuracy.eval(feed_dict={x: newtest_imagehalf, y_: newtest_labelhalf, keep_prob: 1.0})
print("testing accuracy %g" % (test_accuracy))
plt.figure(1)
plt.plot(train_time, losschart)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()

plt.figure(2)
plt.plot(accuracy_time, accuracychart)
plt.xlabel("Iteration, every 50 step")
plt.ylabel("Training Accuracy")
plt.grid()
plt.show()
plt.figure(3)
plt.plot(accuracy_time, test_accuracy_chart)
plt.xlabel("Iteration, every 50 step")
plt.ylabel("Testing Accuracy")
plt.grid()
plt.show()


  #train_step.run(feed_dict={x: images_batch_val, y_: labels_batch_val, keep_prob: 0.5})
#print("test accuracy %g"%accuracy.eval(feed_dict={x: newtrain_image, y_: newtrain_label, keep_prob: 1.0}))


