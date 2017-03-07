import tensorflow as tf
import load_minst
import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
hidden_units = 300
train_image = load_minst.load_mnist("training", np.arange(10) ,"/home/yuming/cogs181/hw4")[0]
train_label = load_minst.load_mnist("training", np.arange(10) ,"/home/yuming/cogs181/hw4")[1]
test_image = load_minst.load_mnist("testing", np.arange(10) ,"/home/yuming/cogs181/hw4")[0]
test_label = load_minst.load_mnist("testing", np.arange(10) ,"/home/yuming/cogs181/hw4")[1]
newtest_image = test_image.reshape(10000,784)
newtest_label = np.zeros([10000, 10])
#one hot encoding
for i in range(0,len(test_label)):
    newtest_label[i,test_label[i]] = 1

newtrain_image = train_image.reshape(60000,784)
newtrain_label = np.zeros([60000, 10])

#one hot encoding
for i in range(0,len(train_label)):
    newtrain_label[i,train_label[i,0]] = 1
#print("one hot encoing", newtrain_label)
images_batch = tf.placeholder(dtype = tf.float32, shape = [784])
labels_batch = tf.placeholder(dtype = tf.float32, shape = [10])

weights = {
    'c': tf.Variable(tf.random_normal([784, 300])),
    #'c': tf.Variable(tf.random_normal([784, 10], 1, 1)),
    'w': tf.Variable(tf.random_normal([300, 10]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([300])),
    'b2': tf.Variable(tf.random_normal([10]))
}
print("weights of c : ", weights['c'])

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
def multiplayer_perceptron(x, weights, biases):
    layer1 = tf.matmul(x, weights['c']) + biases['b1']
    layer1 = tf.sigmoid(layer1)
    output = tf.matmul(layer1, weights['w']) + biases['b2']

    return output

y = multiplayer_perceptron(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
train_step = tf.train.AdamOptimizer(learning_rate=0.002).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
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
training_epochs = 150
avg_cost = 0
time = 0
display_step = 1
losschart = np.zeros(training_epochs)
train_time = np.arange(training_epochs)
training_accuracy_chart = np.zeros(training_epochs)
testing_accuracy_chart = np.zeros(training_epochs)
#calculate accu
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
for epoch in range(training_epochs):
    avg_cost = 0
    #print("shape: ", newtrain_image.shape[0])
    total_batch = int(newtrain_image.shape[0]/batch_size)
    for i in range(total_batch):

        #print(time)
        time += 1
        images_batch_val, labels_batch_val = next(iter_)
        #print("image batch val : ", images_batch_val)
        #print("labels batch val : ", labels_batch_val)
        _, loss_val = sess.run([train_step,cross_entropy], feed_dict = {x: images_batch_val, y_: labels_batch_val})
        #print("loss is : ", loss_val)
        avg_cost += loss_val / total_batch
    if epoch % display_step == 0:
        print("ith: ", epoch)
        print("Epoch:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(avg_cost))
        losschart[epoch] = avg_cost

        training_accuracy = accuracy.eval({x: newtrain_image, y_: newtrain_label})
        testing_accuracy = accuracy.eval({x: newtest_image, y_: newtest_label})
        training_accuracy_chart[epoch] = training_accuracy
        testing_accuracy_chart[epoch] = testing_accuracy
        print("Training Accuracy:", training_accuracy)
        print("Test Accuracy:", testing_accuracy)

print("Optimization Finished!")
plt.figure(1)
plt.plot(train_time, losschart)
plt.grid()

plt.figure(2)
plt.plot(train_time, training_accuracy_chart)
plt.xlabel("Training iterations")
plt.ylabel("Training accuracy")
plt.grid()

plt.figure(3)
plt.plot(train_time, testing_accuracy_chart)
plt.xlabel("Training iterations")
plt.ylabel("Testing accuracy")
plt.grid()
plt.show()

print("Final Test Accuracy:", accuracy.eval({x: newtest_image, y_: newtest_label}))