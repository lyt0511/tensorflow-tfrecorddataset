#coding:utf-8
"""Routine for decoding the CIFAR-10 or CIFAR-100 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
from six.moves import urllib
import tarfile

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
# IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.

#用于描述CiFar数据集的全局常量
# NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_DEPTH = 3
NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR20 = 20
NUM_CLASSES_CIFAR100 = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

print('调用我啦...cifar_input...')

#从网址下载数据集存放到data_dir指定的目录下
CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR100_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

#从网址下载数据集存放到data_dir指定的目录中
def maybe_download_and_extract(data_dir,data_url=CIFAR10_DATA_URL):
    """下载并解压缩数据集 from Alex's website."""
    dest_directory = data_dir #'../CIFAR10_dataset'
    DATA_URL = data_url
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1] #'cifar-10-binary.tar.gz'
    filepath = os.path.join(dest_directory, filename)#'../CIFAR10_dataset\\cifar-10-binary.tar.gz'
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if data_url== CIFAR10_DATA_URL:
        extracted_dir_path = os.path.join(dest_directory,'cifar-10-batches-bin')  # '../CIFAR10_dataset\\cifar-10-batches-bin'
    else :
        extracted_dir_path = os.path.join(dest_directory, 'cifar-100-binary')  # '../CIFAR10_dataset\\cifar-10-batches-bin'
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_cifar10(filename_queue,coarse_or_fine=None):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.

  #cifar10 binary中的样本记录：3072=32x32x3
  #<1 x label><3072 x pixel>
  #...
  #<1 x label><3072 x pixel>

  # 类型标签字节数
  label_bytes = 1  # 2 for CIFAR-100

  result.height = 32
  result.width = 32
  result.depth = 3

    #图像字节数
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  # 每一条样本记录由 标签 + 图像 组成，其字节数是固定的。
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  # 创建一个固定长度记录读取器，读取一个样本记录的所有字节（label_bytes + image_bytes)
  # 由于cifar10中的记录没有header_bytes 和 footer_bytes,所以设置为0
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,header_bytes=0,footer_bytes=0)

  # 调用读取器对象的read 方法返回一条记录
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  #将一个字节组成的string类型的记录转换为长度为record_bytes，类型为unit8的一个数字向量
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  # 将一个字节代表了标签，我们把它从unit8转换为int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # 剩余的所有字节都是图像数据，把他从unit8转换为int32
  # 转为三维张量[depth，height，width]
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  # 把图像的空间位置和深度位置顺序由[depth, height, width] 转换成[height, width, depth]
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def read_cifar100(filename_queue,coarse_or_fine='fine'):
  """Reads and parses examples from CIFAR100 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  class CIFAR100Record(object):
    pass
  result = CIFAR100Record()
  result.height = 32
  result.width = 32
  result.depth = 3

  # cifar100中每个样本记录都有两个类别标签，每一个字节是粗略分类标签，
  # 第二个字节是精细分类标签：<1 x coarse label><1 x fine label><3072 x pixel>
  coarse_label_bytes = 1
  fine_label_bytes = 1

  #图像字节数
  image_bytes = result.height * result.width * result.depth

  # 每一条样本记录由 标签 + 图像 组成，其字节数是固定的。
  record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes

  # 创建一个固定长度记录读取器，读取一个样本记录的所有字节（label_bytes + image_bytes)
  # 由于cifar100中的记录没有header_bytes 和 footer_bytes,所以设置为0
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,header_bytes=0,footer_bytes=0)

  # 调用读取器对象的read 方法返回一条记录
  result.key, value = reader.read(filename_queue)

  #将一系列字节组成的string类型的记录转换为长度为record_bytes，类型为unit8的一个数字向量
  record_bytes = tf.decode_raw(value, tf.uint8)

  # 将一个字节代表了粗分类标签，我们把它从unit8转换为int32.
  coarse_label = tf.cast(tf.strided_slice(record_bytes, [0], [coarse_label_bytes]), tf.int32)

  # 将二个字节代表了细分类标签，我们把它从unit8转换为int32.
  fine_label = tf.cast(tf.strided_slice(record_bytes, [coarse_label_bytes], [coarse_label_bytes + fine_label_bytes]), tf.int32)

  if coarse_or_fine == 'fine':
    result.label = fine_label #100个精细分类标签
  else:
    result.label = coarse_label #100个粗略分类标签

  # 剩余的所有字节都是图像数据，把他从一维张量[depth * height * width]
  # 转为三维张量[depth，height，width]
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [coarse_label_bytes + fine_label_bytes],
                       [coarse_label_bytes + fine_label_bytes + image_bytes]),
                        [result.depth, result.height, result.width])

  # 把图像的空间位置和深度位置顺序由[depth, height, width] 转换成[height, width, depth]
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(cifar10or20or100,data_dir, batch_size):
  """使用Reader ops 构造distorted input 用于CIFAR的训练

  输入参数:
   cifar10or20or100:指定要读取的数据集是cifar10 还是细分类的cifar100 ，或者粗分类的cifar100
    data_dir: 指向CIFAR-10 或者 CIFAR-100 数据集的目录
    batch_size: 每个批次的图像数量

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  #判断是读取cifar10 还是 cifar100（cifar100可分为20类或100类）
  if cifar10or20or100 == 10:
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in xrange(1,6)]
    read_cifar = read_cifar10
    coarse_or_fine = None
  if cifar10or20or100 == 20:
    filenames = [os.path.join(data_dir,'train.bin')]
    read_cifar = read_cifar100
    coarse_or_fine = 'coarse'
  if cifar10or20or100 == 100:
      filenames = [os.path.join(data_dir, 'train.bin')]
      read_cifar = read_cifar100
      coarse_or_fine = 'fine'

  #检查文件是否存在
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # 根据文件名列表创建一个文件名队列
  filename_queue = tf.train.string_input_producer(filenames)

  # 从文件名队列的文件中读取样本
  read_input = read_cifar(filename_queue)

  # 将无符号8位图像数据转换成float32位
  casted_image = tf.cast(read_input.uint8image, tf.float32)

  # 要生成的目标图像的大小，在这里与原图像的尺寸保持一致
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  #为图像添加padding = 4，图像尺寸变为[32+4,32+4],为后面的随机裁切留出位置
  padded_image = tf.image.resize_image_with_crop_or_pad(casted_image,width+4,height+4)

  #下面的这些操作为原始图像添加了很多不同的distortions，扩增了原始训练数据集

  # 在[36,36]大小的图像中随机裁切出[height,width]即[32,,32]的图像区域
  distorted_image = tf.random_crop(padded_image, [height, width, 3])

  # 将图像进行随机的水平翻转（左边和右边的像素对调）
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # 下面这两个操作不满足交换律，即 亮度调整+对比度调整 和 对比度调整+亮度调整
  # 产生的结果是不一样的，你可以采取随机的顺序来执行这两个操作
  distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)

  # 数据集标准化操作：减去均值+方差归一化(divide by the variance of the pixels)
  float_image = tf.image.per_image_standardization(distorted_image)

  # 设置张量的形状
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # 确保： the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def inputs(cifar10or20or100, eval_data, data_dir, batch_size):
  """使用Reader ops 读取数据集，用于CIFAR的评估

  输入参数:
  cifar10or20or100:指定要读取的数据集是cifar10 还是细分类的cifar100 ，或者粗分类的cifar100
    eval_data: True or False ,指示要读取的是训练集还是测试集
    data_dir: 指向CIFAR-10 或者 CIFAR-100 数据集的目录
    batch_size: 每个批次的图像数量

  返回:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  #判断是读取cifar10 还是 cifar100（cifar100可分为20类或100类）
  if cifar10or20or100 == 10:
      read_cifar = read_cifar10
      coarse_or_fine = None
      if not eval_data:
          filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in xrange(1,6)]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
      else:
          filenames = [os.path.join(data_dir,'test_batch.bin')]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  if cifar10or20or100 == 20 or cifar10or20or100 == 100:
      read_cifar = read_cifar100
      if not eval_data:
          filenames = [os.path.join(data_dir,'train.bin')]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
      else:
          filenames = [os.path.join(data_dir,'test.bin')]
          num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  if cifar10or20or100 == 100:
      coarse_or_fine = 'fine'
  if cifar10or20or100 == 20:
      coarse_or_fine = 'coarse'

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # 根据文件名列表创建一个文件名队列
  filename_queue = tf.train.string_input_producer(filenames)

  # 从文件名队列的文件中读取样本
  read_input = read_cifar(filename_queue, coarse_or_fine = coarse_or_fine)
  # 将无符号8位图像数据转换成float32位
  casted_image = tf.cast(read_input.uint8image, tf.float32)

  # 要生成的目标图像的大小，在这里与原图像的尺寸保持一致
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # 用于评估过程的图像数据预处理
  # Crop the central [height, width] of the image.（其实这里并未发生裁剪）
  resized_image = tf.image.resize_image_with_crop_or_pad(casted_image,width,height)

  #数据集标准化操作：减去均值 + 方差归一化
  float_image = tf.image.per_image_standardization(resized_image)

  # 设置数据集中张量的形状
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  # 通过构造样本队列(a queue of examples)产生一个批次的图像和标签
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)