# -*- coding:utf-8 -*-
import tensorflow as tf
import random
import sys
import os
import time

'''tfrecord 写入数据.
将图片数据写入 tfrecord 文件。以 MNIST png格式数据集为例。

首先将图片解压到 ../../cifar100_test/ 目录下。
解压以后会有 training 和 testing 两个数据集。在每个数据集下，有100个文件夹，分别存放了这100个类别的数据。
每个文件夹名为对应的类别编码。

现在网上关于打包图片的例子非常多，实现方式各式各样，效率也相差非常多。
选择合适的方式能够有效地节省时间和硬盘空间。
有几点需要注意：
1.打包 tfrecord 的时候，千万不要使用 Image.open() 或者 matplotlib.image.imread() 等方式读取。
 1张小于10kb的png图片，前者（Image.open) 打开后，生成的对象100+kb, 后者直接生成 numpy 数组，大概是原图片的几百倍大小。
 所以应该直接使用 tf.gfile.FastGFile() 方式读入图片。
2.从 tfrecord 中取数据的时候，再用 tf.image.decode_png() 对图片进行解码。
3.不要随便使用 tf.image.resize_image_with_crop_or_pad 等函数，可以直接使用 tf.reshape()。前者速度极慢。
4.如果有固态硬盘的话，图片数据一定要放在固态硬盘中进行读取，速度能高几十倍几十倍几十倍！生成的 tfrecord 文件就无所谓了，找个机械盘放着就行。
'''

# 图片文件路径
TRAINING_DIR = '/media/cloud/Files/dataset/cifar100/cifar100_tfrecord/cifar100_train/'
TESTING_DIR = '/media/cloud/Files/dataset/cifar100/cifar100_tfrecord/cifar100_test/'
# tfrecord 文件保存路径,这里只保存一个 tfrecord 文件
TRAINING_TFRECORD_NAME = 'training.tfrecord'
TESTING_TFRECORD_NAME = 'testing.tfrecord'

# 把 label(文件名) 转为对应 id
DICT_LABEL_TO_ID = {}
for i in range(100):
    DICT_LABEL_TO_ID[str(i)] = i



def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def convert_tfrecord_dataset_single(dataset_dir, tfrecord_name, tfrecord_path='data/'):
    """ convert samples to tfrecord dataset.
        single版本产生单个tfrecord文件
    Args:
        dataset_dir: 数据集的路径。
        tfrecord_name: 保存为 tfrecord 文件名
        tfrecord_path: 保存 tfrecord 文件的路径。
    """
    if not os.path.exists(dataset_dir):
        print(u'png文件路径错误，请检查是否已经解压png文件。')
        exit()
    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(os.path.dirname(tfrecord_path))
    tfrecord_file = os.path.join(tfrecord_path, tfrecord_name)
    class_names = os.listdir(dataset_dir)
    n_class = len(class_names)
    print(u'一共有 %d 个类别' % n_class)
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for class_name in class_names:  # 对于每个类别
            class_dir = os.path.join(dataset_dir, class_name)  # 获取类别对应的文件夹路径
            file_names = os.listdir(class_dir)  # 在该文件夹下，获取所有图片文件名
            label_id = DICT_LABEL_TO_ID.get(class_name)  # 获取类别 id
            print(u'\n正在处理类别 %d 的数据' % label_id)
            time0 = time.time()
            n_sample = len(file_names)
            for i in range(n_sample):
                file_name = file_names[i]
                sys.stdout.write('\r>> Converting image %d/%d , %g s' % (
                    i + 1, n_sample, time.time() - time0))
                png_path = os.path.join(class_dir, file_name)  # 获取每个图片的路径
                # CNN inputs using
                img = tf.gfile.FastGFile(png_path, 'rb').read()  # 读入图片
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': bytes_feature(img),
                            'label': int64_feature(label_id)
                        }))
                serialized = example.SerializeToString()
                writer.write(serialized)
    print('\nFinished writing data to tfrecord files.')

def convert_tfrecord_dataset_multi(dataset_dir, tfrecord_name, tfrecord_path='data_m/'):
    """ convert samples to tfrecord dataset.
        multi版本产生多个tfrecord文件
    Args:
        dataset_dir: 数据集的路径。
        tfrecord_name: 保存为 tfrecord 文件名
        tfrecord_path: 保存 tfrecord 文件的路径。
    """
    if not os.path.exists(dataset_dir):
        print(u'png文件路径错误，请检查是否已经解压png文件。')
        exit()
    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(os.path.dirname(tfrecord_path))
    class_names = os.listdir(dataset_dir)
    n_class = len(class_names)
    print(u'一共有 %d 个类别' % n_class)

    num_files=100

    for i in range(num_files):
        print("write ",i," file")
        fileName=("-%.5d-of-%.5d" % (i,num_files))
        fileName=tfrecord_path+tfrecord_name+fileName
        writer=tf.python_io.TFRecordWriter(path=fileName)

        # 对于每个类别
        class_dir = os.path.join(dataset_dir, str(i))  # 获取类别对应的文件夹路径
        file_names = os.listdir(class_dir)  # 在该文件夹下，获取所有图片文件名
        label_id = DICT_LABEL_TO_ID.get(str(i))  # 获取类别 id
        print(u'\n正在处理类别 %d 的数据' % label_id)
        time0 = time.time()
        n_sample = len(file_names)
        for j in range(n_sample):
            file_name = file_names[j]
            sys.stdout.write('\r>> Converting image %d/%d , %g s' % (
                j + 1, n_sample, time.time() - time0))
            png_path = os.path.join(class_dir, file_name)  # 获取每个图片的路径
            # CNN inputs using
            img = tf.gfile.FastGFile(png_path, 'rb').read()  # 读入图片
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': bytes_feature(img),
                        'label': int64_feature(label_id)
                    }))
            serialized = example.SerializeToString()
            writer.write(serialized)
        writer.close()

    print('\nFinished writing data to tfrecord files.')


'''read data
从 tfrecord 文件中读取数据，对应数据的格式为png / jpg 等图片数据。
'''

def pares_tf(example_proto):
    # **3.根据你写入的格式对应说明读取的格式
    features = tf.parse_single_example(example_proto,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       }
                                       )
    img = features['image']
    # 这里需要对图片进行解码
    img = tf.image.decode_jpeg(img, channels=3)  # 这里，也可以解码为 1 通道
    # 图片归一化，如果不做归一化，结果会很难看
    # 如果image数据类型已是float，再使用convert_image_dtype转成float类型，并不会对image的数值做归一化
    # 则需通过别的途径进行归一化处理，如：(image-mean(image,0))/var(image,0)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    img = tf.reshape(img, [32, 32, 3])  # 28*28*3
    tf.transpose(img, [2, 1, 0])

    label = features['label']
    label = tf.cast(label,tf.int32)
    #label = tf.one_hot(label, depth=100, on_value=1)

    return img, label

'''
#读取PIL或者cv方式写入图片的tfrecord
def pares_tf_PIL(example_proto):
    # **3.根据你写入的格式对应说明读取的格式
    features = tf.parse_single_example(example_proto,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       }
                                       )
    img = features['image']
    # 这里需要对图片进行解码
    img = tf.decode_raw(img, out_type=tf.uint8)
    img = tf.reshape(img, [32, 32, 3])  # 28*28*3
    tf.transpose(img, [2, 1, 0])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    label = features['label']
    label = tf.cast(label,tf.int32)
    #label = tf.one_hot(label, depth=100, on_value=1)

    return img, label
'''

#读取单个tfrecord
def dataset_read_single(filenames=['/home/cloud/tensorflow/study/dataset_create/data/training.tfrecord'],buffer_size=50000,batch_size=128):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(pares_tf)
    #repeat写在shuffle之后，而不是读取batch之后
    #我们要做的是对整个dataset repeat多次，以供多个epoch使用
    #如果在batch后repeat就仅仅是对当前batch repeat，这样当本次data数据流读取完毕，就会报错
    #因为如果data_num/batch_num不是整数的话，最后一个batch肯定小于设定的batch_size
    shuffle_dataset = dataset.shuffle(buffer_size=buffer_size).repeat()
    shuffle_dataset = shuffle_dataset.batch(batch_size)


    iterator = shuffle_dataset.make_one_shot_iterator()

    return iterator

#读取多个tfrecord，filenames是待读取tfrecord的文件名队列
def dataset_read_multi(filenames=['data/training.tfrecord','data/training.tfrecord']):
    random.shuffle(filenames)
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(pares_tf)
    shuffle_dataset = dataset.shuffle(buffer_size=50000).repeat()
    shuffle_dataset = shuffle_dataset.batch(100)


    iterator = shuffle_dataset.make_one_shot_iterator()

    return iterator
