#coding:utf-8

import cifar10_input
import tfrecord_dataset
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 50000
train_samples_num = 50000
eval_samples_num = 10000
batch_size_tr = 128
batch_size_ts = 125
cifar_data_dir = '/media/cloud/Files/dataset/cifar-100-binary/cifar-100-python/'
tfrecord_data_dir_tr = ['/home/cloud/tensorflow/study/dataset_create/data/training.tfrecord']
tfrecord_data_dir_ts = ['/home/cloud/tensorflow/study/dataset_create/data/testing.tfrecord']

image_holder_tr = tf.placeholder(tf.float32, [batch_size_tr, 32, 32, 3])
label_holder_tr = tf.placeholder(tf.int32, [batch_size_tr])
image_holder_ts = tf.placeholder(tf.float32, [batch_size_ts, 32, 32, 3])
label_holder_ts = tf.placeholder(tf.int32, [batch_size_ts])

'''
#读取cifar数据集，使用tensorflow的cifar_input方法
#distorted_inputs函数产生训练需要使用的数据，包括特征和其对应的label,
#返回已经封装好的tensor，每次执行都会生成一个batch_size的数量的样本
'''
def cifar_input(cifar_type=100, eval_data=False, data_dir=cifar_data_dir, batch_size=batch_size_tr):

    if eval_data == 'False':
        images, labels = cifar10_input.distorted_inputs(cifar10or20or100=cifar_type,
                                                        data_dir=data_dir,
                                                        batch_size=batch_size)
    else:
        images, labels = cifar10_input.inputs(cifar10or20or100=cifar_type,
                                              eval_data=eval_data,
                                              data_dir=data_dir,
                                              batch_size=batch_size)

    return images, labels


'''
#读取自己生成的tfrecord记录的数据集
'''
def tfrecord_input(data_dir, buffer_size=train_samples_num, batch_size=batch_size_tr):
    iterator = tfrecord_dataset.dataset_read_single(data_dir, buffer_size, batch_size)
    next_element = iterator.get_next()
    return next_element


def variable_with_weight_loss(shape, stddev, wl):
    '''定义初始化weight函数,使用tf.truncated_normal截断的正态分布，但加上L2的loss，相当于做了一个L2的正则化处理'''
    var = tf.get_variable(name='weight', initializer=tf.truncated_normal(shape, stddev=stddev))
    '''w1:控制L2 loss的大小，tf.nn.l2_loss函数计算weight的L2 loss'''
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        '''tf.add_to_collection:把weight losses统一存到一个collection，名为losses'''
        tf.add_to_collection('losses', weight_loss)

    return var

def model(X, batch_size):

    '''第一个卷积层：使用variable_with_weight_loss函数创建卷积核的参数并进行初始化。
    第一个卷积层卷积核大小：5x5 3：颜色通道 64：卷积核数目
    weight1初始化函数的标准差为0.05，不进行正则wl(weight loss)设为0'''
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
        # tf.nn.conv2d函数对输入image_holder进行卷积操作
        kernel1 = tf.nn.conv2d(X, weight1, [1, 1, 1, 1], padding='SAME')

        bias1 = tf.get_variable(name='bias', initializer=tf.constant(0.0, shape=[64]))

        conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    # 最大池化层尺寸为3x3,步长为2x2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # LRN层模仿生物神经系统的'侧抑制'机制
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        '''第二个卷积层：'''
        weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
        kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
        # bias2初始化为0.1
        bias2 = tf.get_variable(name='bias', initializer=tf.constant(0.1, shape=[64]))

        conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
        weight3 = variable_with_weight_loss(shape=[dim, 4096], stddev=0.04, wl=0.004)
        bias3 = tf.get_variable(name='bias', initializer=tf.constant(0.1, shape=[4096]))
        local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

    # 全连接层，隐含层节点数下降了一半
    with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
        weight4 = variable_with_weight_loss(shape=[4096, 4096], stddev=0.04, wl=0.004)
        bias4 = tf.get_variable(name='bias', initializer=tf.constant(0.1, shape=[4096]))
        local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

    with tf.variable_scope('fc3', reuse=tf.AUTO_REUSE):
        '''正态分布标准差设为上一个隐含层节点数的倒数，且不计入L2的正则'''
        weight5 = variable_with_weight_loss(shape=[4096, 100], stddev=1 / 4096.0, wl=0.0)
        bias5 = tf.get_variable(name='bias', initializer=tf.constant(0.0, shape=[100]))
        logits = tf.add(tf.matmul(local4, weight5), bias5)

    return logits


'''
计算CNN的loss
tf.nn.sparse_softmax_cross_entropy_with_logits作用：
把softmax计算和cross_entropy_loss计算合在一起
'''
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    # tf.reduce_mean对cross entropy计算均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
    tf.add_to_collection('losses', cross_entropy_mean)
    # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

'''
训练时每次循环需要的优化操作以及预测时的top准确度

将logits节点和label_holder传tt入loss计算得到最终loss
'''
def one_step_train(img, lb, batch_size):
    logits = model(img, batch_size)
    losses = loss(logits, lb)

    train_op = tf.train.AdamOptimizer(1e-4).minimize(losses)

    return train_op, losses

def one_step_eval(img, lb, batch_size):
    logits = model(img, batch_size)
    losses = loss(logits, lb)
    # 求输出结果中top k的准确率，默认使用top 1(输出分类最高的那一类的准确率)
    top_k_op = tf.nn.in_top_k(logits, lb, 1)

    return top_k_op

def train(dataset_type='tfrecord'):
    images_train, labels_train = cifar_input()
    images_test, labels_test = cifar_input(cifar_type=100, eval_data=True, data_dir=cifar_data_dir, batch_size=batch_size_ts)
    next_element_tr = tfrecord_input(tfrecord_data_dir_tr,train_samples_num,batch_size_tr)
    next_element_ts = tfrecord_input(tfrecord_data_dir_ts,eval_samples_num,batch_size_ts)
    train_op, losses = one_step_train(image_holder_tr, label_holder_tr, batch_size_tr)
    top_k_op = one_step_eval(image_holder_ts, label_holder_ts, batch_size_ts)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    steps_per_epoch = int(train_samples_num / batch_size_tr) + 1
    steps_cur = 1
    epochs_cur = 1

    for step in range(max_steps):
        '''training:'''
        start_time = time.time()
        # 获得一个batch的训练数据
        if dataset_type=='cifar':
            image_batch_tr, label_batch_tr = sess.run([images_train, labels_train])
        elif dataset_type=='tfrecord':
            image_batch_tr, label_batch_tr = sess.run(fetches=next_element_tr)
        # 将batch的数据传入train_op和loss的计算
        _, loss_value = sess.run([train_op, losses],
                                 feed_dict={image_holder_tr: image_batch_tr, label_holder_tr: label_batch_tr})

        duration = time.time() - start_time

        if steps_cur % 101 == 0 or steps_cur == 1:
            # 每秒能训练的数量
            samples_per_sec = batch_size_tr / duration
            # 一个batch数据所花费的时间
            sec_per_batch = float(duration)

            format_str = ('epochs %d, steps %d/%d, loss=%.2f (%.1f samples/sec; %.3f sec/batch)')
            print(format_str % (epochs_cur, steps_cur, steps_per_epoch, loss_value, samples_per_sec, sec_per_batch))

        if epochs_cur % 10 == 0 and steps_cur == 1:
            num_iter = int(math.ceil(eval_samples_num / batch_size_ts))
            true_count = 0
            total_sample_count = num_iter * batch_size_ts
            step_eval = 0
            while step_eval < num_iter:
                # 获取images-test labels_test的batch
                if dataset_type == 'cifar':
                    image_batch_ts, label_batch_ts = sess.run([images_test, labels_test])
                elif dataset_type == 'tfrecord':
                    image_batch_ts, label_batch_ts = sess.run(fetches=next_element_ts)
                # 计算这个batch的top 1上预测正确的样本数
                preditcions = sess.run([top_k_op], feed_dict={image_holder_ts: image_batch_ts,
                                                              label_holder_ts: label_batch_ts
                                                              })
                # 全部测试样本中预测正确的数量
                true_count += np.sum(preditcions)
                step_eval = step_eval + 1
            # 准确率
            precision = true_count / total_sample_count
            print('test precision @ 1 = %.3f' % precision)

        if steps_cur == steps_per_epoch+1:
            steps_cur = 1
            epochs_cur = epochs_cur + 1
        else:
            steps_cur = steps_cur + 1


def eval(dataset_type='tfrecord'):
    images_test, labels_test = cifar_input(cifar_type=100, eval_data=True, data_dir=cifar_data_dir,
                                           batch_size=batch_size_ts)
    next_element_ts = tfrecord_input(tfrecord_data_dir_ts, eval_samples_num, batch_size_ts)
    top_k_op = one_step_eval(image_holder_ts, label_holder_ts, batch_size_ts)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    # 样本数
    num_iter = int(math.ceil(eval_samples_num / batch_size_ts))
    true_count = 0
    total_sample_count = num_iter * batch_size_ts
    step_eval = 0
    while step_eval < num_iter:
        # 获取images-test labels_test的batch
        if dataset_type == 'cifar':
            image_batch_ts, label_batch_ts = sess.run([images_test, labels_test])
        elif dataset_type == 'tfrecord':
            image_batch_ts, label_batch_ts = sess.run(fetches=next_element_ts)
        # 计算这个batch的top 1上预测正确的样本数
        preditcions = sess.run([top_k_op], feed_dict={image_holder_ts: image_batch_ts,
                                                      label_holder_ts: label_batch_ts
                                                      })
        # 全部测试样本中预测正确的数量
        true_count += np.sum(preditcions)
        step_eval = step_eval + 1
    # 准确率
    precision = true_count / total_sample_count
    print('The final test precision @ 1 = %.3f' % precision)


train()
eval()