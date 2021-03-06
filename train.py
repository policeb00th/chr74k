import preprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse

class Classifier():
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0):
        """Defining the model."""

        self.scope = scope
        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.input = tf.placeholder(tf.float32, [None, img_h, img_w, 1])

        self.conv1 = slim.conv2d(
                self.input,
                num_outputs=32, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv1'
        )
        self.conv2 = slim.conv2d(
                self.conv1,
                num_outputs=128, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv2'
        )
        self.conv3 = slim.conv2d(
                self.conv2,
                num_outputs=128, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv3'
        )
        self.pool1 = slim.max_pool2d(self.conv3, [2, 2])
        self.conv4 = slim.conv2d(
                self.pool1,
                num_outputs=256, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv4'
        )
        self.conv5 = slim.conv2d(
                self.conv4,
                num_outputs=256, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv5'
        )
        self.pool2 = slim.max_pool2d(self.conv5, [2, 2])
        self.conv6 = slim.conv2d(
                self.pool2,
                num_outputs=512, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv6'
        )
        self.conv7 = slim.conv2d(
                self.conv6,
                num_outputs=512, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv7'
        )
        self.conv8 = slim.conv2d(
                self.conv7,
                num_outputs=512, kernel_size=[3, 3],
                stride=[1, 1], padding='Valid',
                scope=self.scope+'_conv8'
        )
        self.pool = slim.max_pool2d(self.conv8, [2, 2])


        self.hidden = slim.fully_connected(
                slim.flatten(self.pool),
                8192,
                scope=self.scope+'_hidden',
                activation_fn=tf.nn.relu
        )
        self.classes1 = slim.fully_connected(
                self.hidden,
                4096,
                scope=self.scope+'_fc1',
                activation_fn=tf.nn.relu
        )
        self.classes2 = slim.dropout(self.classes1)
        self.classes3 = slim.fully_connected(
                self.classes2,
                4096,
                scope=self.scope+'_fc2',
        )
        self.classes4 = slim.dropout(self.classes3)
        self.classes = slim.fully_connected(
                self.classes4,
                self.n_classes,
                scope=self.scope+'_fc3',
                activation_fn=None
        )
        self.targets = tf.placeholder(tf.int32, [None])
        self.targets_onehot = tf.one_hot(self.targets, self.n_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.targets_onehot,
                logits=self.classes
        ))
        self.train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)


def train(model_name, training_dataset, validation_dataset):
    img_h, img_w = 64, 64
    train_steps = int(1e5)
    batch_size = 10

    nn = Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES), 0.5)
    dataset = list(map(lambda f:f.strip(),
                       open(training_dataset, 'r').readlines()))
    validation_dataset = list(map(lambda f:f.strip(), 
                                  open(validation_dataset, 'r').readlines()))

    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('summaries/'+model_name)

        for t in range(train_steps):
            
            # perform training step
            images, labels = preprocessing.get_batch(dataset, 10, (img_h, img_w))
            loss, _ = sess.run([nn.loss, nn.train_step], feed_dict={
                nn.input   : images,
                nn.targets : labels
            })

            # show and save training status
            if t % 10 == 0: print(t, loss)
            if t % 1000 == 0: saver.save(sess, 'saves/'+model_name, global_step=t)

            summary = tf.Summary()
            summary.value.add(tag='Loss', simple_value=float(loss))
            if t % 50 == 0:
                # testing model on validation set occasionally
                images, labels = preprocessing.get_batch(
                        validation_dataset, 20, (img_h, img_w))
                classes = sess.run(nn.classes, feed_dict={nn.input:images})
                summary.value.add(tag='ValidationError',
                        simple_value=float(sum(np.argmax(classes, -1) != labels)))
            summary_writer.add_summary(summary, t)
            summary_writer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-t', type=str, required=True, help='Training dataset name')
    parser.add_argument(
            '-v', type=str, required=True, help='Validation dataset name')
    parser.add_argument('-m', type=str, required=True, help='Model name')

    opt = parser.parse_args()
train(opt.m, opt.t, opt.v)