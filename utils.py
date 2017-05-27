"""
    Author : Byunghyun Ban
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.01.
"""

import re
import tensorflow as tf
import matplotlib.pyplot as plt
import Evaluator as ev
import datetime
import numpy as np


def get_conv_shape(name):
    spec = re.split(':|, |->', name)
    kernel_size = int(spec[5])
    stride = int(spec[7])
    input_fm = int(spec[9])
    output_fm = int(spec[10])
    conv_shape = [kernel_size, kernel_size, input_fm, output_fm]
    return conv_shape, stride


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d(x, W, bias, stride=1):
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def deconv(x, W, b, output_shape, stride=1):
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def change_format(image):
    return ((image * 255) / np.max(image)).astype('uint8')


def save_images(batch_size, directory, input_image, output_image, ground_truth, itr, show_image=False):

    label = 'PSNR: {:.2f} SSIM: {:.2f}'

    for i in range(batch_size):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()
        visual_psnr = ev.psnr(1, ground_truth[i], input_image[i])
        visual_predict_psnr = ev.psnr(1, ground_truth[i], output_image[i])
        visual_ssim = ev.ssim(1, ground_truth[i], input_image[i])
        visual_predict_ssim = ev.ssim(1, ground_truth[i], output_image[i])


        ax[0].imshow(change_format(input_image[i,:,:,0]))
        ax[0].set_xlabel(label.format(visual_psnr, visual_ssim))
        ax[0].set_title('Input Image')
        ax[1].imshow(change_format(output_image[i,:,:,0]))
        ax[1].set_xlabel(label.format(visual_predict_psnr, visual_predict_ssim))
        ax[1].set_title('Output Image')
        ax[2].imshow(change_format(ground_truth[i,:,:,0]))
        ax[2].set_title('Ground Truth')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        
        time = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        
        fig.savefig("logs/images/%s__%d.jpeg" % (time, itr))

        if show_image is True:
            plt.show()
        plt.cla()
        plt.close(fig)


