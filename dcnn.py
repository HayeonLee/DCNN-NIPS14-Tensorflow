
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import tensorflow as tf
import Datareader as dr
import Datareader as dr2
import Evaluator as ev
import utils
import datetime
import numpy as np


training_data_dir = "images/train/"
validation_data_dir = "images/validation/"
logs_dir = "logs"

IMAGE_RESIZE = int(1)
IMAGE_SIZE = 256
MAX_ITERATION = int(300000)
GT_RESIZE = int(1)
learning_rate = 1e-3


stddev = 0.02

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("tr_batch_size", "4", "batch size for training. [default : 5]")
tf.flags.DEFINE_integer("val_batch_size", "4", "batch size for validation. [default : 5]")
tf.flags.DEFINE_bool("reset", "True", "mode : True or False [default : True]")
tf.flags.DEFINE_string('mode',"train", "mode : train/ test/ visualize/ evaluation [default : train]")
tf.flags.DEFINE_string("device", "/gpu:0", "device : /cpu:0, /gpu:0 [default : /gpu:0]")


if FLAGS.reset:
    print('** Note : directory was reset! **')
    if 'win32' in sys.platform:
        os.popen('rmdir /s /q' + logs_dir)
    else:
        os.popen('rm -rf ' + logs_dir + "/*")
        os.popen('rm -rf ' + logs_dir)

    os.popen('mkdir ' + logs_dir)
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/valid')
    os.popen('mkdir ' + logs_dir + '/train/psnr')
    os.popen('mkdir ' + logs_dir + '/train/loss_g')
    os.popen('mkdir ' + logs_dir + '/train/loss_d')
    os.popen('mkdir ' + logs_dir + '/valid/psnr')
    os.popen('mkdir ' + logs_dir + '/valid/loss_g')
    os.popen('mkdir ' + logs_dir + '/valid/loss_d')
    os.popen('mkdir ' + logs_dir + '/images')
    os.popen('mkdir ' + logs_dir + '/visualize_result')
    os.popen('mkdir ' + logs_dir + '/images/train')



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(conv, b)


def deconv2d(x, W, output_shape, b):
    deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(deconv, b)

class DCNN :
    def __init__(self, batch_size, is_training=True):
        self.is_training = is_training
        self.low_resolution_image = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE*IMAGE_RESIZE, IMAGE_SIZE*IMAGE_RESIZE, 3], name="low_resolution_image")
        self.high_resolution_image = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE*IMAGE_RESIZE, IMAGE_SIZE*IMAGE_RESIZE, 3], name="high_resolution_image")

        self.image_R, _,_ = tf.split(self.high_resolution_image, num_or_size_splits=3, axis=3)

        self.predict_y = self.DCNN_graph(self.low_resolution_image, is_training=is_training)

        self.loss = tf.reduce_mean(tf.squared_difference(self.predict_y, self.image_R))

        trainable_var = tf.trainable_variables()

        #self.train_optimizer = self.train(trainable_var)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=trainable_var)

        #def train(self, trainable_var):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
    #    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=trainable_var)
     #   return optimizer

    def conv(self, input_feature , kernel_shape, bias_shape, strides_shape=[1, 1, 1, 1], padding="SAME"):
        # Create variable named "weights"
        weights = tf.get_variable(name="weights", initializer=tf.truncated_normal(kernel_shape, stddev=stddev))
        # Create variable named "bias"
        biases = tf.get_variable(name="bias", initializer=tf.constant(0.0, shape=bias_shape))
        conv_val = tf.nn.conv2d(input_feature, weights, strides=strides_shape, padding=padding)
        return conv_val, biases

    def conv_tanh(self, input_feature , kernel_shape, bias_shape, strides_shape=[1, 1, 1, 1], padding="SAME"):
        conv_val, biases = self.conv(input_feature, kernel_shape, bias_shape, strides_shape=strides_shape, padding=padding)
        return tf.tanh(conv_val + biases)

    def DCNN_graph(self, image, is_training):
        self.is_training = is_training

        #image_R, image_G, image_B = tf.split(image, num_or_size_splits=3, axis=3)
        ###### Deconvolution Sub-Network ######
        # Layer 1_1 : kernel size : 1*121, bias : 38, input channel : 1, output channel : 38
        W_conv1 = weight_variable([3, 121, 1, 38])
        b_conv1 = bias_variable([38])
        #h_conv1 = tf.nn.relu(conv2d(image_R, W_conv1,b_conv1))
        h_conv1_R = tf.nn.relu(conv2d(image_R, W_conv1, b_conv1))
        h_conv1_G = tf.nn.relu(conv2d(image_G, W_conv1, b_conv1))
        h_conv1_B = tf.nn.relu(conv2d(image_B, W_conv1, b_conv1))

        # Layer 1_2 : kernel size : 121*1, bias : 38, input channel : 38, output channel : 38
        W_conv2 = weight_variable([121, 1, 38, 38])
        b_conv2 = bias_variable([38])
        #h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, b_conv2))
        h_conv2_R = tf.nn.relu(conv2d(h_conv1_R, W_conv2, b_conv2))
        h_conv2_G = tf.nn.relu(conv2d(h_conv1_G, W_conv2, b_conv2))
        h_conv2_B = tf.nn.relu(conv2d(h_conv1_B, W_conv2, b_conv2))

        ###### Outlier Rejection Sub-Network ######
        # Layer 2_1 : kernel size : 16*16, bias : 512, input channel : 38, output channel : 512
        W_conv3 = weight_variable([16, 16, 38, 512])
        b_conv3 = bias_variable([512])
        #h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, b_conv3))
        h_conv3_R = tf.nn.relu(conv2d(h_conv2_R, W_conv3, b_conv3))
        h_conv3_G = tf.nn.relu(conv2d(h_conv2_G, W_conv3, b_conv3))
        h_conv3_B = tf.nn.relu(conv2d(h_conv2_B, W_conv3, b_conv3))

        # Layer 2_2 : kernel size : 1*1, bias : 512, input channel : 512, output channel : 512
        W_conv4 = weight_variable([1, 1, 512, 512])
        b_conv4 = bias_variable([512])
        #h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, b_conv4))
        h_conv4_R = tf.nn.relu(conv2d(h_conv3_R, W_conv4, b_conv4))
        h_conv4_G = tf.nn.relu(conv2d(h_conv3_G, W_conv4, b_conv4))
        h_conv4_B = tf.nn.relu(conv2d(h_conv3_B, W_conv4, b_conv4))

        ###### Restoration ######
        # Layer 3 : Deconv(Transpose conv) of kernel size : 8*8, bias : 512, input channel : 1, output channel : 512
        # Use conv function (Same) : kernel size : 8*8, bias : 1, input channel : 512, output channel : 1
        W_conv5 = weight_variable([8, 8, 512, 1])
        b_conv5 = bias_variable([1])
        #o_conv5 = conv2d(h_conv4, W_conv5, b_conv5)
        o_conv5_R = conv2d(h_conv4_R, W_conv5, b_conv5)
        o_conv5_G = conv2d(h_conv4_G, W_conv5, b_conv5)
        o_conv5_B = conv2d(h_conv4_B, W_conv5, b_conv5)
        # Use deconv function
        # W_dconv5 = weight_variable([8, 8, 512, 1])
        # b_dconv5 = bias_variable([1])
        # o_conv5_R = tf.tanh(dconv2d(h_conv4_R, W_dconv5, output_shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 1], b_dconv5))
        # o_conv5_G = tf.tanh(dconv2d(h_conv4_G, W_dconv5, output_shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 1], b_dconv5))
        # o_conv5_B = tf.tanh(dconv2d(h_conv4_B, W_dconv5, output_shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 1], b_dconv5))

        predict_y = tf.concat([o_conv5_R, o_conv5_G, o_conv5_B], 3)
        #predict_y = o_conv5
        return predict_y

def train(is_training=True):
    ###############################  GRAPH PART  ###############################
    print("Graph Initialization...")
    with tf.device(FLAGS.device):
        with tf.variable_scope("model") as scope:
            m_train = DCNN(FLAGS.tr_batch_size, is_training=True)
	    scope.reuse_variables()
        #with tf.variable_scope("model", reuse=True): # reuse trained parameters above (name: model/..:0)
            m_valid = DCNN(FLAGS.val_batch_size, is_training=False)
    print("Done")

    ##############################  Summary Part  ##############################
    print("Setting up summary op...")
    loss = tf.placeholder(dtype=tf.float32)
    loss_summary_op = tf.summary.scalar("LOSS", loss)
    psnr = tf.placeholder(dtype=tf.float32)
    psnr_summary_op = tf.summary.scalar("PSNR", psnr)
    ssim = tf.placeholder(dtype=tf.float32)
    ssim_summary_op = tf.summary.scalar("SSIM", ssim)

    train_summary_writer_loss = tf.summary.FileWriter(logs_dir + '/train/loss', max_queue=2)
    valid_summary_writer_loss = tf.summary.FileWriter(logs_dir + '/valid/loss', max_queue=2)
    train_summary_writer_psnr = tf.summary.FileWriter(logs_dir + '/train/psnr', max_queue=2)
    valid_summary_writer_psnr = tf.summary.FileWriter(logs_dir + '/valid/psnr', max_queue=2)
    train_summary_writer_ssim = tf.summary.FileWriter(logs_dir + '/train/ssim', max_queue=2)
    valid_summary_writer_ssim = tf.summary.FileWriter(logs_dir + '/valid/ssim', max_queue=2)

    print("Done")

    ############################  Model Save Part  #############################
    #print("Setting up Saver...")
    #saver = tf.train.Saver()
    #ckpt = tf.train.get_checkpoint_state(logs_dir)
    #print("Done")

    ################################  Datareader Part  ################################
    print("Datareader Initialization...")
    validation_dataset_reader = dr2.Dataset(path=validation_data_dir,
                                           input_shape=(int(IMAGE_SIZE*IMAGE_RESIZE), int(IMAGE_SIZE*IMAGE_RESIZE)),
                                           gt_shape=(int(IMAGE_SIZE*GT_RESIZE), int(IMAGE_SIZE*GT_RESIZE)))
    if FLAGS.mode == "train":
        train_dataset_reader = dr.Dataset(path=training_data_dir,
                                          input_shape=(int(IMAGE_SIZE * IMAGE_RESIZE), int(IMAGE_SIZE * IMAGE_RESIZE)),
                                          gt_shape=(int(IMAGE_SIZE * GT_RESIZE), int(IMAGE_SIZE * GT_RESIZE)))
    print("Done")

    ################################  Optimization Part  ################################
    print("Otimizer Initialization...")
    #with tf.variable_scope("sleep") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(m_train.loss)
    #reuse_variables()
    #optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(m_valid.loss)
    print("Done")

    ################################  Session Part  ################################
    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    print("Model restored...")
    #else:
    sess.run(tf.global_variables_initializer())
    print("Done")

    #############################     Train      ###############################
    if FLAGS.mode == "train":
        for itr in range(MAX_ITERATION):
            train_low_resolution_image, train_high_resolution_image = train_dataset_reader.next_batch(FLAGS.tr_batch_size)
            train_dict = {m_train.low_resolution_image: train_low_resolution_image,
                          m_train.high_resolution_image: train_high_resolution_image}

            #sess.run([m_train.train_optimizer], feed_dict=train_dict)
            sess.run([optimizer], feed_dict=train_dict)
            #sess.run([optimizer2], feed_dict=)
            if itr % 10 == 0:
                valid_low_resolution_image, valid_high_resolution_image = validation_dataset_reader.next_batch(FLAGS.val_batch_size)
                valid_dict = {m_valid.low_resolution_image: valid_low_resolution_image,
                              m_valid.high_resolution_image: valid_high_resolution_image}
                train_loss, train_predict_y = sess.run([m_train.loss, m_train.predict_y], feed_dict=train_dict)
                valid_loss, valid_predict_y = sess.run([m_valid.loss, m_valid.predict_y], feed_dict=valid_dict)
                train_high_resolution_image = train_high_resolution_image.astype(np.uint8)
                train_predict_y = train_predict_y.astype(np.uint8)
                valid_high_resolution_image = valid_high_resolution_image.astype(np.uint8)
                valid_predict_y = valid_predict_y.astype(np.uint8)

                #image_R,_,_ = np.split(train_high_resolution_image,3, axis=3)
                #image_R2,_,_ = np.split(valid_high_resolution_image,3, axis=3)

                train_psnr, train_ssim = ev.evaluator(FLAGS.tr_batch_size, train_high_resolution_image, train_predict_y)
                valid_psnr, valid_ssim = ev.evaluator(FLAGS.val_batch_size, valid_high_resolution_image, valid_predict_y)
                tr_loss_str, tr_psnr_str, tr_ssim_str = sess.run([loss_summary_op, psnr_summary_op, ssim_summary_op], feed_dict={loss: train_loss, psnr: train_psnr, ssim: train_ssim})
                valid_loss_str, valid_psnr_str, valid_ssim_str = sess.run([loss_summary_op, psnr_summary_op, ssim_summary_op], feed_dict={loss: valid_loss, psnr: valid_psnr, ssim: valid_ssim})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_summary_writer_loss.add_summary(tr_loss_str,itr)
                valid_summary_writer_loss.add_summary(valid_loss_str, itr)
                train_summary_writer_psnr.add_summary(tr_psnr_str,itr)
                valid_summary_writer_psnr.add_summary(valid_psnr_str, itr)
                train_summary_writer_ssim.add_summary(tr_ssim_str,itr)
                valid_summary_writer_ssim.add_summary(valid_ssim_str, itr)

                print("%s ---> Validation_PSNR: %g" % (datetime.datetime.now(), valid_psnr))
                print("%s ---> Validation_SSIM: %g" % (datetime.datetime.now(), valid_ssim))
                print("Step: %d, Train_mean_PSNR:%g" % (itr, train_psnr))
                print("Step: %d, Train_mean_SSIM:%g" % (itr, train_ssim))

            #if itr % 50 == 0:
            #    saver.save(sess, logs_dir + "/model.ckpt", itr)

            if itr % 50 == 0:
                visual_low_resolution_image, visual_high_resolution_image = validation_dataset_reader.random_batch(
                    FLAGS.val_batch_size)
                visual_dict = {m_valid.low_resolution_image: visual_low_resolution_image,
                                   m_valid.high_resolution_image: visual_high_resolution_image}
                predict = sess.run(m_valid.predict_y, feed_dict=visual_dict)
                #image_r,_,_ = np.split(visual_high_resolution_image,3,axis=3)
                #image_r2,_,_ = np.split(visual_low_resolution_image,3,axis=3)

                utils.save_images(FLAGS.val_batch_size, logs_dir + '/images', visual_high_resolution_image, predict,
                                  visual_low_resolution_image, itr, show_image=False)
                print('Validation images were saved!')
                utils.save_images(FLAGS.tr_batch_size, logs_dir + '/images/train', train_low_resolution_image, train_predict_y,
                                  train_high_resolution_image, itr, show_image=False)
                print('Train images were saved!')
    ###########################     Visualize     ##############################
    elif FLAGS.mode == "visualize":

        visual_low_resolution_image, visual_high_resolution_image = validation_dataset_reader.random_batch(FLAGS.val_batch_size)
        visual_dict = {m_valid.low_resolution_image: visual_low_resolution_image,
                               m_valid.high_resolution_image: visual_high_resolution_image}
        #predict = sess.run(m_valid.predict_y, feed_dict=visual_dict)
        #utils.save_images(FLAGS.val_batch_size, validation_data_dir, visual_low_resolution_image, predict,
        #                          visual_high_resolution_image,0, show_image=False)
        print('Validation images were saved!')


def main():
    train(True)
    #pass

main()

'''
def train(is_training=True):

    # Create the model
    low_resolution_image = tf.placeholder(tf.float32, shape=[FLAGS.tr_batch_size,
                                                             IMAGE_SIZE * IMAGE_RESIZE,
                                                             IMAGE_SIZE * IMAGE_RESIZE,
                                                             3], name="low_resolution_image")
    high_resolution_image = tf.placeholder(tf.float32, shape=[FLAGS.tr_batch_size,
                                                              IMAGE_SIZE * IMAGE_RESIZE,
                                                              IMAGE_SIZE * IMAGE_RESIZE,
                                                              3], name="high_resolution_image")
    # Import data
    train_dataset_reader = dr.Dataset(path=training_data_dir,
                                      input_shape=(IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE),
                                      gt_shape=(IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE))

    for i in range(FLAGS.tr_batch_size):
        predict_y = DCNN.DCNN_graph(low_resolution_image[i], is_training)
        loss = tf.reduce_mean(tf.square(predict_y - high_resolution_image[i]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)




        #cross_entropy = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(label=high_resolution_image, logits=y_conv))
   # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    ##############################  Summary Part  ##############################
    print("Setting up summary op...")
    loss_log = tf.placeholder(dtype=tf.float32)
    loss_log_summary_op = tf.summary.scalar("Loss", loss_log)
    psnr_log = tf.placeholder(dtype=tf.float32)
    psnr_log_summary_op = tf.summary.scalar("PSNR", psnr_log)

    train_summary_writer_loss = tf.summary.FileWriter(logs_dir + '/train/loss', max_queue=2)
    train_summary_writer_psnr = tf.summary.FileWriter(logs_dir + '/train/psnr', max_queue=2)
    #graph_summary_writer_loss = tf.summary.FileWriter(logs_dir + '/train/loss_graph', max_queue=2)
    print("Done")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for itr in range(MAX_ITERATION):
            train_low_resolution_image, train_high_resolution_image = train_dataset_reader.next_batch(FLAGS.tr_batch_size, 32)

            _, predict_loss, predict_y = sess.run([train_step, loss, y_conv],
                                                  feed_dict={low_resolution_image: train_low_resolution_image,
                                                             high_resolution_image: train_high_resolution_image})

            if itr % 10 == 0:
                train_summary_loss_str = sess.run(loss_log_summary_op, feed_dict={loss_log: predict_loss})
                train_summary_writer_loss.add_summary(train_summary_loss_str, itr)
                train_psnr = ev.psnr(FLAGS.tr_batch_size, train_high_resolution_image, predict_y)
                train_summary_psnr_str = sess.run(psnr_log_summary_op, feed_dict={psnr_log: train_psnr})
                train_summary_writer_psnr.add_summary(train_summary_psnr_str, itr)
                train_summary_writer_loss.add_graph(sess.graph)
                #graph_summary_writer_loss
                print("Step : %d, loss : %g" % (itr, predict_loss))
                print("Step : %d, PSNR : %g" % (itr, train_psnr))

            if itr % 500 == 0:
                #print(train_high_resolution_image.dtype)
                #print(train_low_resolution_image.dtype)
                #print(predict_y.dtype)

                utils.save_images(FLAGS.tr_batch_size, training_data_dir+"result", train_low_resolution_image,
                                  predict_y, train_high_resolution_image, show_image_num=None)
                print('images were saved!')
'''


'''
image_R , image_G, image_B = tf.split(low_resolution_image, num_or_size_splits=3, axis=3)

###### Deconvolution Sub-Network ######
# Layer 1_1 : kernel size : 1*121, bias : 38, input channel : 1, output channel : 38
W_conv1 = weight_variable([1, 121, 1, 38])
b_conv1 = bias_variable([38])
#h_conv1 = tf.nn.relu(conv2d(low_resolution_image, W_conv1) + b_conv1)
h_conv1_R = tf.tanh(conv2d(image_R, W_conv1, b_conv1))
h_conv1_G = tf.tanh(conv2d(image_G, W_conv1, b_conv1))
h_conv1_B = tf.tanh(conv2d(image_B, W_conv1, b_conv1))

# Layer 1_2 : kernel size : 121*1, bias : 38, input channel : 38, output channel : 38
W_conv2 = weight_variable([121, 1, 38, 38])
b_conv2 = bias_variable([38])
h_conv2_R = tf.tanh(conv2d(h_conv1_R, W_conv2, b_conv2))
h_conv2_G = tf.tanh(conv2d(h_conv1_G, W_conv2, b_conv2))
h_conv2_B = tf.tanh(conv2d(h_conv1_B, W_conv2, b_conv2))

###### Outlier Rejection Sub-Network ######
# Layer 2_1 : kernel size : 16*16, bias : 512, input channel : 38, output channel : 512
W_conv3 = weight_variable([16, 16, 38, 512])
b_conv3 = bias_variable([512])
h_conv3_R = tf.tanh(conv2d(h_conv2_R, W_conv3, b_conv3))
h_conv3_G = tf.tanh(conv2d(h_conv2_G, W_conv3, b_conv3))
h_conv3_B = tf.tanh(conv2d(h_conv2_B, W_conv3, b_conv3))

# Layer 2_2 : kernel size : 1*1, bias : 512, input channel : 512, output channel : 512
W_conv4 = weight_variable([1, 1, 512, 512])
b_conv4 = bias_variable([512])
h_conv4_R = tf.tanh(conv2d(h_conv3_R, W_conv4, b_conv4))
h_conv4_G = tf.tanh(conv2d(h_conv3_G, W_conv4, b_conv4))
h_conv4_B = tf.tanh(conv2d(h_conv3_B, W_conv4, b_conv4))

###### Restoration ######
# Layer 3 : Deconv(Transpose conv) of kernel size : 8*8, bias : 512, input channel : 1, output channel : 512
# Use conv function (Same) : kernel size : 8*8, bias : 1, input channel : 512, output channel : 1
W_conv5 = weight_variable([8, 8, 512, 1])
b_conv5 = bias_variable([1])
o_conv5_R = tf.tanh(conv2d(h_conv4_R, W_conv5, b_conv5))
o_conv5_G = tf.tanh(conv2d(h_conv4_G, W_conv5, b_conv5))
o_conv5_B = tf.tanh(conv2d(h_conv4_B, W_conv5, b_conv5))
# Use deconv function
#W_dconv5 = weight_variable([8, 8, 512, 1])
#b_dconv5 = bias_variable([1])
#o_conv5_R = tf.tanh(dconv2d(h_conv4_R, W_dconv5, output_shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 1], b_dconv5))
#o_conv5_G = tf.tanh(dconv2d(h_conv4_G, W_dconv5, output_shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 1], b_dconv5))
#o_conv5_B = tf.tanh(dconv2d(h_conv4_B, W_dconv5, output_shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 1], b_dconv5))

y_conv = tf.concat([o_conv5_R, o_conv5_G, o_conv5_B], 3)
'''


'''
class DCNN :
    def __init__(self, batch_size, is_training=True):
        self.low_resolution_image = tf.placeholder(tf.float32, [batch_size,
                                                                IMAGE_SIZE*IMAGE_RESIZE,
                                                                IMAGE_SIZE*IMAGE_RESIZE,
                                                                3], name="low_resolution_image")
        self.high_resolution_image = tf.placeholder(tf.float32, [batch_size,
                                                                 IMAGE_SIZE*IMAGE_RESIZE,
                                                                 IMAGE_SIZE*IMAGE_RESIZE,
                                                                 3], name="high_resolution_image")
        train_dataset_reader = dr.Dataset(path=training_data_dir,
                                          input_shape=(IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE),
                                          gt_shape=(IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE))
        image_R , image_G, image_B = tf.split(self.low_resolution_image, num_or_size_splits=3, axis=3)
'''
