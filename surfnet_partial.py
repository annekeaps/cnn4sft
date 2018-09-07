from sys import executable
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import tensorflow as tf
import os
import skimage.io
from skimage.color import rgb2gray
import cv2
from itertools import cycle
from collections import namedtuple
from data import *
from tensor_cv import *
from bilinear import *
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

class cnn4sft(object):

    def __init__(self, mode):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.object_name='forest'
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.batch_size = 1
        self.num_threads = 8

        # START
        self.mode = mode
        self.home =  '/home/anneke/project/surf/tensorflow/'

        self.num_channels = 1 #gray image
        self.fc_size = 1024

        self.height = 128
        self.width = 128
        self.height_f = 128.
        self.width_f = 128.

        # Camera parameter
        # --- Calibrate the Camera
        param_internal = [[1004.39, 0., 654.779], [0., 1004.39, 360.319], [0.,0.,1.]]
        param_internal = np.asarray(param_internal)

        # --- Scale
        size = 128.
        calib_height = 720
        calib_width = 1280

        sx = size/float(calib_width)
        sy = size/float(calib_height)

        scale_mtx = np.zeros((3,3))
        scale_mtx[0][0] = sx
        scale_mtx[1][1] = sy
        scale_mtx[2][2] = 1.0

        self.camera_mtx = np.matmul(scale_mtx, param_internal)

        if self.mode == 'train':
            self.train()
        else :
            self.test()


    def network_preparation(self):
        if self.mode == 'train':
            self.home_img = self.home+'data/train/'
        else:
            self.batch_size = 1
            #self.home_img = self.home+'data/test/'
            self.home_img = self.home+'data/train/'

        # Checkpoint Path
        self.retrain = False
        #self.checkpoint_path = '/home/anneke/project/surf/tensorflow/model/-1000'
        self.checkpoint_path = ''
        self.log_directory = '/home/anneke/project/surf/tensorflow/model/'

        # txt file of filenames data
        self.filenames_file = self.home + 'data/'+ self.mode +'_txt/forest.txt'

        # Load data
        data = LoadData(self.home_img, self.filenames_file, \
            self.num_threads, self.batch_size, self.height, self.width, self.mode)

        self.img_input = data.image_batch

        # Load Template
        forest_path = '/home/anneke/project/surf/tensorflow/data/objects/forest/'

        texture = cv2.imread(forest_path+"texture.jpg")
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
        self.size_template = texture.shape
        texture = texture / 255.
        texture = texture.astype(np.float32)

        self.text_flat = np.reshape(texture, (-1))

        # Load image element files
        self.vertices_norm = self.load_file(file_path=forest_path, filename='vertices_norm', num_fields=4)
        vertices = self.load_file(file_path=forest_path, filename='vertices', num_fields=4)
        vertices = vertices.astype(np.int32)
        self.edges = self.load_file(file_path=forest_path, filename='edges', num_fields=3)
        self.edges = self.edges.astype(np.int32)
        mesh = self.load_file(file_path=forest_path, filename='faces', num_fields=4)
        mesh = mesh.astype(np.int32)

        self.vertices = vertices

        # face filling
        # Faces with its fillpoints (2D index) -- [x y id_face]
        template_np_fullxy = self.load_file(file_path=forest_path, filename='face_filling', num_fields=3)
        self.template_np_fullxy = template_np_fullxy.astype('float32')

        size_mesh = mesh.shape
        self.n_mesh = size_mesh[0]

        # Linking face with its points
        edges_gather = mesh[:,1:]
        edges_gather = np.reshape(edges_gather, (-1))

        vert = self.edges[edges_gather,1:]
        vert = np.reshape(vert, (-1))
        self.vert = vert

        vertices_gather=[]
        for i in range(0, len(vert), 6):
            a = np.unique(vert[i:i+6])
            vertices_gather.append(a)

        vertices_gather = np.asarray(vertices_gather)
        vertices_gather = np.reshape(vertices_gather, [-1])

        # gathered vertices(id) in order by edges and faces -- [v1, v2, ...]
        self.template_np_gathervert = vertices_gather

        # x and y
        points_gather = vertices[vertices_gather,1:3]

        # Gathered xy in order by faces (2D index) -- [xy]
        points_gather = points_gather.astype(np.float32)
        self.template_np_gatherxy = points_gather

    def getLoss(self):
        return self.total_loss

    def count_text_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines)

    def load_file(self, file_path, filename, num_fields):
        points = np.loadtxt(file_path + filename + '.txt')

        if num_fields==3:
            a,b,c = zip(*points)
            list = np.column_stack((a, b, c))
            list.astype(np.int32)

        elif num_fields==4:
            a,b,c,d = zip(*points)
            list = np.column_stack((a, b, c, d))
            #list.astype(np.float32)

        return list

    def test(self):
        with tf.Graph().as_default():
            # Network preparation
            self.network_preparation()

            with tf.variable_scope(tf.get_variable_scope()):
                # Build Network
                self.build_net()

                with tf.name_scope('concat-xyz'):
                    # Collect the 3D geometry data estimation
                    x = self.conv2[0,:,:,0]
                    num_elements = x.shape[0]*x.shape[1]
                    x = tf.reshape(x, [num_elements, 1])

                    y = self.conv2[0,:,:,1]
                    y = tf.reshape(y, [num_elements, 1])

                    z = self.conv2[0,:,:,2]
                    z = tf.reshape(z, [num_elements, 1])

                    xyz = tf.concat([x, y, z], 1)
                    self.xyz = xyz

                # Sampling the object points
                #idx_xyz = np.linspace(0, 4095, (self.col+1)*(self.row+1))
                #idx_xyz = idx_xyz.astype(np.int32)
                """
                idx_xyz = np.random.randint(4096, size=16)

                with tf.name_scope('sampling-points'):
                    xyz_sample = tf.gather(xyz, idx_xyz)
                    self.xyz_sample = xyz_sample

                # Compute Deformation Loss
                loss_deform, t, im = self.compute_loss_deformation(xyz_sample)
                """

                #"""
                with tf.name_scope('project-points'):
                    # Project the points
                    #xyz = tf.multiply(tf.add(xyz, 1.), 0.5)
                    #self.xyz = xyz
                    xyz = tf.transpose(tf.cast(xyz, tf.float64))
                    xyz_2d = tf.matmul(self.camera_mtx, xyz)
                    xyz_2d = tf.cast(xyz_2d, tf.float32)
                    xyz_2d = tf.transpose(xyz_2d)
                    self.xyz_2d = xyz_2d

                    x = tf.divide(xyz_2d[:,0], xyz_2d[:,2])
                    y = tf.divide(xyz_2d[:,1], xyz_2d[:,2])

                    x1 = tf.reshape(xyz_2d, [-1,1])
                    y1 = tf.reshape(xyz_2d, [-1,1])

                    xy = tf.concat([x1, y1], 1)
                    self.xy_2d = xy
                #"""
                # Compute Deformation Loss
                loss_deform, t, im, check = self.compute_loss_deformation(self.xyz)

                # Apply the threshold
                #self.xy = tf.clip_by_value(xy, 0., 127.)
                #"""
                """
                ##########
                # Synthetic Data
                test_path = '/home/anneke/project/surf/tensorflow/data/objects/tes/'
                vertices = self.load_file(file_path=test_path, filename='vertices', num_fields=4)
                vertices = tf.convert_to_tensor(vertices)
                self.xy = vertices[:, 1:3]
                ##########
                """

                #appearance = self.generate_appearance()

                #self.write_xyz(x=x, y=y, z=z, path=self.home, filename='coord')

            # SESSION
            config = tf.ConfigProto(allow_soft_placement=True)
            sess = tf.Session(config=config)

            # SAVER
            train_saver = tf.train.Saver()

            # INIT
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            # RESTORE
            if self.checkpoint_path == '':
                restore_path = tf.train.latest_checkpoint(self.log_directory)
            else:
                restore_path = self.checkpoint_path.split(".")[0]

            #train_saver = tf.train.import_meta_graph(restore_path + '.meta')
            train_saver.restore(sess, restore_path)

            ##############
            # -----------------------Show Images-----------------------
            xyz, loss, t, im, check, xy, xyy = sess.run([self.xyz, loss_deform, t, im, check, self.xyz_2d, self.xy_2d])
            #"""
            print 'loss\t: ', loss
            print 'template : ', t
            print 'pred\t: ', im
            print check

            #"""

            self.write_xyz(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], path=self.home, filename='coord')
            self.write_xyz(x=xy[:,0], y=xy[:,1], z=xy[:,2], path=self.home, filename='coord_3d')
            self.write_xy(x=xyy[:,0], y=xyy[:,1], path=self.home, filename='coord_2d')

            """
            [app, xy, img_input]= sess.run([appearance, self.xy, self.img_input], feed_dict={self.texture_flat: self.text_flat, \
                self.template_fullxy: self.template_np_fullxy, self.template_xy: self.template_np_gatherxy, self.idx_vert: self.template_np_gathervert })

            # Appearance Result
            templatexy, objxy, app_t, app_o, app_obj_input = app

            img_input = np.reshape(img_input, (self.height, self.width))

            templatexy = templatexy.astype(np.int32)
            objxy = objxy.astype(np.int32)
            """
            """
            # print object points
            print '\nxy\n', app_obj_input, '\n\n'
            self.write_xyz(x=objxy[:,0], y=objxy[:,1], path=self.home, filename='coord')
            """
            """
            # show template and new object appearance
            im_template = np.ones([self.size_template[1], self.size_template[0]],np.float32)
            new_img = np.ones([self.width, self.height],np.float32)
            obj_input = np.ones([self.width, self.height],np.float32)

            # Template Appearance
            for i in range(0, len(templatexy)):
                im_template[templatexy[i,0], templatexy[i,1]] = app_t[i]

            # Object Appearance Estimation
            for i in range(0, len(objxy)):
                new_img[objxy[i,0], objxy[i,1]] = app_o[i]

            # Object Appearance in Input Image
            for i in range(0, len(objxy)):
                obj_input[objxy[i,0], objxy[i,1]] = app_obj_input[i]


            #-----------------show object appearance estimation------------------------
            f = plt.figure(1)
            plt.imshow(new_img.T, cmap='gray')
            plt.scatter(x=xy[:,0], y=xy[:,1], c='r', s=10)

            # Show mesh
            for i in range (0,len(self.edges)):
                x=np.zeros(2)
                y=np.zeros(2)
                vertex1 = self.edges[i][1]
                vertex2 = self.edges[i][2]
                #print '\nvertex: ',vertex1,'\t', vertex2

                x[0] = xy[vertex1][0]
                y[0] = xy[vertex1][1]
                x[1] = xy[vertex2][0]
                y[1] = xy[vertex2][1]

                plt.plot(x,y, 'ro-')

            #-------------show object appearance in the input image--------------------
            g = plt.figure(2)
            plt.imshow(img_input, cmap='gray')
            #plt.scatter(x=xy[:,0], y=xy[:,1], c='r', s=10)

            # Show mesh
            for i in range (0,len(self.edges)):
                x=np.zeros(2)
                y=np.zeros(2)
                vertex1 = self.edges[i][1]
                vertex2 = self.edges[i][2]

                x[0] = xy[vertex1][0]
                y[0] = xy[vertex1][1]
                x[1] = xy[vertex2][0]
                y[1] = xy[vertex2][1]

                plt.plot(x,y, 'ro-')

            f.show()
            g.show()
            raw_input()


            # -------------------------------------------------
            """
            """
            print tes
            print tes.shape

            x = np.reshape(tes[:,0], (1,-1))
            y = np.reshape(tes[:,1], (1,-1))
            matplotlib.pyplot.scatter(x,y)
            matplotlib.pyplot.show()
            """

    def train(self):
        with tf.Graph().as_default():
            # Network preparation
            self.network_preparation()

            global_step = tf.Variable(0, trainable=False)

            num_training_samples = self.count_text_lines(self.filenames_file)

            steps_per_epoch = np.ceil(num_training_samples / self.batch_size).astype(np.int32)
            num_total_steps = self.num_epochs * steps_per_epoch
            start_learning_rate = self.learning_rate

            boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
            values = [self.learning_rate, self.learning_rate / 2, self.learning_rate / 4]
            self.learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            loss_summary = []

            with tf.variable_scope(tf.get_variable_scope()):
                # Build Network
                self.build_net()

                with tf.name_scope('concat-xyz'):
                    # Collect the 3D geometry data estimation
                    x = self.conv2[0,:,:,0]
                    num_elements = x.shape[0]*x.shape[1]
                    x = tf.reshape(x, [num_elements, 1])

                    y = self.conv2[0,:,:,1]
                    y = tf.reshape(y, [num_elements, 1])

                    z = self.conv2[0,:,:,2]
                    z = tf.reshape(z, [num_elements, 1])

                    xyz = tf.concat([x, y, z], 1)
                    self.xyz = xyz

                # Sampling the object points
                #idx_xyz = np.linspace(0, 4095, (self.col+1)*(self.row+1))
                #idx_xyz = idx_xyz.astype(np.int32)
                """
                idx_xyz = np.random.randint(4096, size=16)

                with tf.name_scope('sampling-points'):
                    xyz_sample = tf.gather(xyz, idx_xyz)
                    self.xyz_sample = xyz_sample
                """
                """
                with tf.name_scope('project-points'):
                    # Project the points
                    xyz = tf.multiply(tf.add(xyz, 1.), 0.5)
                    self.xyz = xyz

                    xyz = tf.transpose(tf.cast(xyz, tf.float64))
                    xyz_2d = tf.matmul(self.camera_mtx, xyz)
                    xyz_2d = tf.cast(xyz_2d, tf.float32)

                    xyz_2d = tf.transpose(xyz_2d)
                    self.xyz_2d = xyz_2d

                    x = tf.divide(xyz_2d[:,0], xyz_2d[:,2])
                    y = tf.divide(xyz_2d[:,1], xyz_2d[:,2])

                    x1 = tf.reshape(x, [-1,1])
                    y1 = tf.reshape(y, [-1,1])

                    xy = tf.concat([x1, y1], 1)
                    self.xy_2d = xy

                    # Apply the threshold
                    #self.xy = tf.clip_by_value(xy, 0., 127.)
                """

                # Generate Appearance
                #loss_app = self.generate_appearance()
                #loss = self.temp
                """
                # Compute Deformation Loss
                loss_deform, t, im = self.compute_loss_deformation(xyz_sample)
                """
                # Compute Deformation Loss

                loss_deform, t, im, check = self.compute_loss_deformation(self.xyz)
                with tf.name_scope('loss'):
                    #loss = tf.add(loss_app, loss_deform)
                    loss = loss_deform
                    loss_summary.append(loss)

                variables = tf.trainable_variables()
                loss_ = optimizer.compute_gradients(loss, variables)

            apply_gradient_op = optimizer.apply_gradients(loss_, global_step=global_step)

            total_loss = tf.reduce_mean(loss_summary)

            tf.summary.scalar('learning_rate', self.learning_rate, ['model_0'])
            tf.summary.scalar('total_loss', total_loss, ['model_0'])
            summary_op = tf.summary.merge_all('model_0')

            # SESSION
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.80
            sess = tf.Session(config=config)

            # SAVER
            summary_writer = tf.summary.FileWriter(self.home + '/cnn4sft', sess.graph)
            train_saver = tf.train.Saver()

            # COUNT PARAMS
            total_num_parameters = 0
            for variable in tf.trainable_variables():
                total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            print("number of trainable parameters: {}".format(total_num_parameters))

            # INIT
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            # LOAD CHECKPOINT IF SET
            if self.checkpoint_path != '':
                train_saver.restore(sess, self.checkpoint_path.split(".")[0])
                if self.retrain:
                    sess.run(global_step.assign(0))

            # START
            start_step = global_step.eval(session=sess)
            start_time = time.time()
            for step in range(start_step, num_total_steps+start_step):
                before_op_time = time.time()
                _,loss_value, t_, im_, check_, weight, xyz, apa = sess.run([apply_gradient_op, loss, t, im, check, self.conv2, self.xyz, self.apa])
                print 't\t: ', t_
                print 'pred\t: ', im_
                print 'loss\t:', loss_value
                print '\n'
                #print weight
                print 'ahaha \n'
                print xyz
                print check_


                """
                _,loss_value = sess.run([apply_gradient_op, loss], feed_dict={\
                    self.texture_flat: self.text_flat, \
                        self.template_fullxy: self.template_np_fullxy, self.template_xy: self.template_np_gatherxy, self.idx_vert: self.template_np_gathervert})
                """
                duration = time.time() - before_op_time
                # 50
                if step and step % 5 == 0:
                    examples_per_sec = self.batch_size / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / step - 1.0) * time_sofar
                    #print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                    #print print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left)
                    """
                    print 'step:', step
                    print 'examples_per_sec: ',examples_per_sec
                    print 'loss_value: ',loss_value
                    print 'time_sofar: ', time_sofar
                    print 'training_time_left: ', training_time_left

                    print 't\t: ', t_
                    print 'pred\t: ', im_
                    print 'loss\t:', loss_value
                    print check_
                    print '\n\n'
                    """
                    summary_str = sess.run(summary_op)

                    """
                    summary_str = sess.run(summary_op, feed_dict={\
                        self.texture_flat: self.text_flat, \
                            self.template_fullxy: self.template_np_fullxy, self.template_xy: self.template_np_gatherxy, self.idx_vert: self.template_np_gathervert})
                    """
                    summary_writer.add_summary(summary_str, global_step=step)

                if step and step % 1000 == 0:
                    train_saver.save(sess, self.home + 'model/', global_step=step)

            train_saver.save(sess, self.home + 'model/', global_step=num_total_steps)

    # Compute Appearance Loss
    def compute_loss_appearance(self, app1, app2):
        with tf.variable_scope('loss-app'):
            La = tf.constant(0)
            diff = tf.subtract(app1, app2)
            loss = tf.square(diff)
            La = tf.reduce_sum(loss)

        return La

    # For 3D
    # Compute Deformation Loss
    def compute_loss_deformation(self, xyz_newObj):
        with tf.variable_scope('loss-deform'):
            # template edges, vertices and points
            points_gather_t = self.vertices_norm[self.vert, 1:]
            #points_gather_t = self.vertices[self.vert, 1:]
            size_p = len(points_gather_t)

            points_gather_t = tf.cast(points_gather_t, tf.float32)

            # new object edges, vertices and points
            points_gather_obj = tf.gather(xyz_newObj, self.vert)
            points_gather_obj = tf.cast(points_gather_obj, tf.float32)
            self.apa = points_gather_obj
            size = tf.size(points_gather_obj)/3

            def distPoints(p):
                delta_x = tf.subtract(p[0,0], p[1,0])
                delta_y = tf.subtract(p[0,1], p[1,1])
                delta_z = tf.subtract(p[0,2], p[1,2])

                dist_xy = tf.add( tf.square(delta_x), tf.square(delta_y) )
                dist_xyz = tf.add( dist_xy, tf.square(delta_z) )

                return dist_xyz
            """
            # For loop
            Ld = tf.constant(0.)

            print 'size: \n', size_p
            for i in range (0, size_p-2) :
                p_t = points_gather_t[i:i+2, :]
                p_obj = points_gather_obj[i:i+2, :]

                distPoints_t = distPoints(p_t)
                distPoints_obj = distPoints(p_obj)

                delta = tf.subtract(distPoints_t, distPoints_obj)
                loss = tf.square(delta)
                Ld = tf.add(Ld, loss)

            """
            def cond(i, Ld, dist_t, dist_pred, check, idx):
                return tf.less(i, size_p)

            def body(i, Ld, dist_t, dist_pred, check, idx):
                p_t = points_gather_t[i:i+2, :]
                p_im = points_gather_obj[i:i+2, :]

                distPoints_t = distPoints(p_t)
                distPoints_im = distPoints(p_im)

                dist_t = tf.add(dist_t, distPoints_t)
                dist_pred = tf.add(dist_pred, distPoints_im)

                delta = tf.subtract(distPoints_t, distPoints_im)
                loss = tf.square(delta)
                Ld = tf.add(Ld, loss)

                temp = tf.stack([distPoints_t, distPoints_im, delta, loss, Ld])
                check = check.write(idx, tf.reshape(temp, [1,-1]))

                return [tf.add(i,2), Ld, dist_t, dist_pred, check, tf.add(idx,1)]

            # initialize
            dist_t = tf.constant(0.)
            dist_pred = tf.constant(0.)
            Ld = tf.constant(0.)
            i = tf.constant(0)
            idx = tf.constant(0)
            check = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True)

            r = tf.while_loop(cond, body, [i, Ld, dist_t, dist_pred, check, idx])

        return r[1], r[2], r[3], r[4].stack()

    # Generate Object Appearance
    def generate_appearance(self):
        #---------------------------DEFINE CUSTOM py_func WITH GRADIENT----------------------------
         # Define custom py_func which takes also a grad op as argument:
        def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

            tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

        # Def custom round function:
        def tf_round(x, name=None):
            with ops.name_scope(name, "round", [x]) as name:
                r = py_func(np_round,
                                [x],
                                [tf.float32],
                                name=name,
                                grad=grad_round)  # <-- here's the call to the gradient
                return r[0]

        def np_round(x):
            a = np.round(x)
            return a.astype(np.float32)

        def grad_round(op, grad):
            return grad

        #------------------------------------------END---------------------------------------------

        with tf.device('/device:GPU:0'):
            # flattened texture
            self.texture_flat = tf.placeholder(dtype=tf.float32, shape=(self.text_flat.shape[0]))

            # fullxy : faces with its fillpoints (2D index) -- [x y id_face]
            self.template_fullxy = tf.placeholder(dtype=tf.float32, shape=(self.template_np_fullxy.shape[0], self.template_np_fullxy.shape[1]))

            # gatherxy : gathered xy in order by edges and faces (2D index) -- [x y]
            self.template_xy = tf.placeholder(dtype=tf.float32, shape=(self.template_np_gatherxy.shape[0], self.template_np_gatherxy.shape[1]))

            # gathervert : gathered vertices(id) in order by edges and faces -- [v1, v2, ...]
            self.idx_vert = tf.placeholder(dtype=tf.int32, shape=(self.template_np_gathervert.shape))

            with tf.name_scope('triangle_vertices'):
                input_flat = tf.reshape(self.img_input, [-1])

                # object gathered xy by edges and faces
                #obj_xy = tf.gather(self.xyz[:,1:3], self.idx_vert)
                obj_xy = tf.gather(self.xy, self.idx_vert)
                obj_xy = tf.cast(obj_xy, tf.float32)

                # 2D into 1D index
                obj_xy1d =  tf.add(tf.multiply(self.width_f, obj_xy[:,1]), obj_xy[:,0])

                def write(points, size, points_, idx_):
                    points_ = points_.write(idx_, points[0])
                    idx_ = tf.add(idx_, 1)

                    points_ = tf.cond(tf.equal(size, 1), lambda: points_.write(idx_, points[0]), lambda: points_.write(idx_, points[1]) )
                    idx_ = tf.add(idx_, 1)

                    points_ = tf.cond(tf.equal(size, 3), lambda: points_.write(idx_, points[2]), lambda: points_.write(idx_, points[0]) )
                    return points_, tf.add(idx_,1)

                def cond(i, idx_, idx, idx_obj):
                    return tf.less(i, tf.size(obj_xy1d))

                def body(i, idx_, idx, idx_obj):
                    points2 = tf.gather(obj_xy1d, idx)

                    # Unique coords
                    obj_xy2, idxx = tf.unique(tf.transpose(tf.reshape(points2, [-1])))
                    idxx, new_idx = tf.unique(tf.transpose(tf.reshape(idxx, [-1])))

                    #obj_xy2 = tf.cast(tf.reshape(obj_xy2, [-1,1]), tf.float32)

                    # object triangle coords
                    #size = tf.size(obj_xy2)
                    #idx_obj, idx_ = write(obj_xy2, size, idx_obj, idx_)

                    size = tf.size(new_idx)
                    idx_obj, idx_ = write(tf.add(new_idx,i), size, idx_obj, idx_)

                    return [tf.add(i,3), idx_, tf.add(idx,3), idx_obj]

                idx_obj = tf.TensorArray(dtype=tf.int32, infer_shape=False, size=1, dynamic_size=True)

                idx = tf.range(3)
                i = tf.constant(0)
                idx_ = tf.constant(0)

                r = tf.while_loop(cond, body, [i, idx_, idx, idx_obj])
                idx_obj_ = tf.reshape(r[3].stack(), [-1])

                # 1D into 2D index
                #obj_x = tf.reshape(tf.cast(tf.floormod(idx_obj_, self.width_f), tf.int32), [-1,1])
                #obj_y = tf.reshape(tf.cast(tf.divide(idx_obj_, self.width_f), tf.int32), [-1,1])
                #obj_xy_ = tf.concat([obj_x, obj_y], 1)

                obj_xy_ = tf.gather(obj_xy, idx_obj_)

                loss_app = tf.constant(0.)
                idx = tf.range(3)

                #------------------------------------------------------------#
                if self.mode == 'test':
                    #Check Output
                    facePoint_t=tf.constant([[0.,0.]])
                    outPoint=tf.constant([[0.,0.]])
                    show_template=tf.constant([0.])
                    show_obj = tf.constant([0.])
                    show_obj_input = tf.constant([0.])
                #-----------------------------------------------------------#
            if self.mode == 'test':
                for n in range(0, self.n_mesh):
                    with tf.name_scope('mtx-affine'):
                        tri_obj = tf.gather(obj_xy_, idx)
                        mtx = affine.getAffineTransform( self.template_xy[i:i+3, :], tri_obj )

                        # FillPoints in a face
                        fill_idx = tf.where(tf.equal(self.template_fullxy[:,0], n))
                        fill_idx = tf.cast(tf.reshape(fill_idx, [-1]), tf.int32)

                    with tf.name_scope('res-mtx'):
                        template_xyface_2d = tf.gather(self.template_fullxy[:,1:], fill_idx)

                        newObj2d = affine.getDst(mtx, template_xyface_2d)
                        newObj2d = tf.transpose(newObj2d)


                    with tf.name_scope('app_template_and_object'):
                        # appearance of template
                        template_xyface =  tf.add(tf.multiply(tf.cast(self.size_template[1], tf.float32), template_xyface_2d[:,1]), template_xyface_2d[:,0])
                        app_template = tf.gather(self.texture_flat, tf.cast(template_xyface, tf.int32))

                        # appearance of new object
                        app_obj = app_template

                    with tf.name_scope('app_input'):
                        # appearance of object in input image
                        #newObj2d = tf_round(newObj2d)
                        newObj2d = tf.clip_by_value(newObj2d, 0., 127.)

                        x = tf.reshape(newObj2d[:,0], [self.batch_size, -1, 1])
                        y = tf.reshape(newObj2d[:,1], [self.batch_size, -1, 1])

                        app_input = bilinear_sampler(self.img_input, x, y)
                        app_input = tf.reshape(app_input, [-1])

                        newObj2d = tf_round(newObj2d)
                        newObj2d = tf.clip_by_value(newObj2d, 0., 127.)

                        temp = self.compute_loss_appearance(app_obj, app_input)
                        loss_app = tf.add(loss_app, temp)
                        a = loss_app

                        idx = tf.add(idx,3)
                        i = i+3

                        #--------------------------------------------------------------------------------------------------#
                        if self.mode == 'test':
                            # check
                            facePoint_t = tf.concat([facePoint_t, template_xyface_2d], 0)   # fill points in template face
                            outPoint = tf.concat([outPoint, newObj2d], 0)             # fill points in new object face
                            show_template = tf.concat([show_template, app_template], 0)     # template appearance
                            show_obj = tf.concat([show_obj, app_obj], 0)              # appearance of new object
                            show_obj_input = tf.concat([show_obj_input, app_input], 0)
                        #--------------------------------------------------------------------------------------------------#

            else :
                with tf.name_scope('gen_app'):
                    def cond1(i, idx, loss_app, n):
                        return tf.less(i, self.n_mesh)

                    def body1(i, idx, loss_app, n):
                        with tf.name_scope('mtx-affine'):
                            tri_obj = tf.gather(obj_xy_, idx)
                            mtx = affine.getAffineTransform( self.template_xy[i:i+3, :], tri_obj )

                            # FillPoints in a face
                            fill_idx = tf.where(tf.equal(self.template_fullxy[:,0], n))
                            fill_idx = tf.cast(tf.reshape(fill_idx, [-1]), tf.int32)

                        with tf.name_scope('res-mtx'):
                            template_xyface_2d = tf.gather(self.template_fullxy[:,1:], fill_idx)

                            newObj2d = affine.getDst(mtx, template_xyface_2d)
                            newObj2d = tf.transpose(newObj2d)

                        with tf.name_scope('app_template_and_object'):
                            # appearance of template
                            template_xyface =  tf.add(tf.multiply(tf.cast(self.size_template[1], tf.float32), template_xyface_2d[:,1]), template_xyface_2d[:,0])
                            app_template = tf.gather(self.texture_flat, tf.cast(template_xyface, tf.int32))

                            # appearance of new object
                            app_obj = app_template

                        with tf.name_scope('app_input'):
                            # appearance of object in input image
                            x = tf.reshape(newObj2d[:,0], [self.batch_size, -1, 1])
                            y = tf.reshape(newObj2d[:,1], [self.batch_size, -1, 1])

                            app_input = bilinear_sampler(self.img_input, x, y)
                            app_input = tf.reshape(app_input, [-1])

                            newObj2d = tf_round(newObj2d)
                            newObj2d = tf.clip_by_value(newObj2d, 0., 127.)

                            temp = self.compute_loss_appearance(app_obj, app_input)
                            loss_app = tf.add(loss_app, temp)
                            a = loss_app

                            idx = tf.add(idx,3)
                            i = i+3

                            #--------------------------------------------------------------------------------------------------#
                            if self.mode == 'test':
                                # check
                                facePoint_t = tf.concat([facePoint_t, template_xyface_2d], 0)   # fill points in template face
                                outPoint = tf.concat([outPoint, newObj2d], 0)             # fill points in new object face
                                show_template = tf.concat([show_template, app_template], 0)     # template appearance
                                show_obj = tf.concat([show_obj, app_obj], 0)              # appearance of new object
                                show_obj_input = tf.concat([show_obj_input, app_input], 0)
                            #--------------------------------------------------------------------------------------------------#

                        return [tf.add(i,3), tf.add(idx,3), loss_app, tf.add(n,1)]

                    # initialize loss appearance cost
                    loss_app = tf.constant(0.)
                    n = tf.constant(0.)
                    i = tf.constant(0)
                    idx = tf.range(3)

                    r = tf.while_loop(cond1, body1, [i, idx, loss_app, n])
                    loss = r[2]
                    #self.temp = a

        if self.mode == 'test':
            return facePoint_t[1:], outPoint[1:], show_template[1:], show_obj[1:], show_obj_input[1:]
        else:
            return loss

    #################
    def build_net(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose]):
            with tf.variable_scope('encoder'):
                # DECODER
                # Shape : [filter_size filter_size num_channels num_output]
                self.conv1, self.weights_conv1 = \
                    self.layer_conv(input=self.img_input, num_channels=self.num_channels, kernel_size=3,
                        num_filters=32, strides=2, padding='SAME', relu=True, bn=False)
                print "Convolution1: " + str(self.conv1.shape)

                num_filters = 64
                self.down_res1= self.down_res_block(input=self.conv1,num_filters=num_filters)

                num_filters = 96
                self.down_res2= self.down_res_block(input=self.down_res1, num_filters=num_filters)

                num_filters = 128
                self.down_res3= self.down_res_block(input=self.down_res2, num_filters=num_filters)

                num_filters = 256
                self.down_res4= self.down_res_block(input=self.down_res3, num_filters=num_filters)

                num_filters = 512
                self.down_res5= self.down_res_block(input=self.down_res4, num_filters=num_filters)

                # FULLY-CONNECTED
                self.flat, num_inputs = self.layer_flatten(self.down_res5)
                self.fc = self.layer_fc(input=self.flat, num_inputs=num_inputs)

            with tf.variable_scope('decoder'):
                # Convolutional Transpose
                num_outputs = 512
                output_size = 2
                self.conv_transpose1 = self.layer_conv_transpose(input=self.fc, kernel_size=1, num_outputs=num_outputs,
                    strides=2, padding='SAME', output_size=output_size, relu=True, bn=False)

                output_size = 4
                self.up_res1 = self.up_res_block(input=self.conv_transpose1, num_outputs=num_outputs, output_size=output_size)

                #"""
                num_outputs = 256
                output_size = 8
                self.up_res2 = self.up_res_block(input=self.up_res1, num_outputs=num_outputs, output_size=output_size)

                num_outputs = 128
                output_size = 16
                self.up_res3 = self.up_res_block(input=self.up_res2, num_outputs=num_outputs, output_size=output_size)

                num_outputs = 96
                output_size = 32
                self.up_res4 = self.up_res_block(input=self.up_res3, num_outputs=num_outputs, output_size=output_size)

                """
                num_outputs = 64
                output_size = 64
                self.up_res5 = self.up_res_block(input=self.up_res4, num_outputs=num_outputs, output_size=output_size)
                """

                # Convolution
                shape = self.up_res4.shape
                num_channels = int(shape[3])
                self.conv2, self.weights_conv2 = \
                    self.layer_conv(input=self.up_res4, num_channels=num_channels, kernel_size=1,
                        num_filters=3, strides=1, padding='VALID', relu=True, bn=False)
                print "Convolution2: " + str(self.conv2.shape)
                """
                # Convolution
                shape = self.up_res1.shape
                num_channels = int(shape[3])
                self.conv2, self.weights_conv2 = \
                    self.layer_conv(input=self.up_res1, num_channels=num_channels, kernel_size=1,
                        num_filters=3, strides=1, padding='VALID', relu=True, bn=False)
                print "Convolution2: " + str(self.conv2.shape)
                """
    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        #return tf.Variable(tf.contrib.layers.xavier_initializer()(shape))

    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def layer_relu(self, layer):
        #return tf.nn.relu(layer)
        return tf.nn.sigmoid(layer)

    def lrelu(self, x, alpha):
        return tf.nn.relu(x) + alpha * tf.nn.relu(-x)

    def layer_bn(self, layer):
        return tf.layers.batch_normalization(layer)

    def layer_flatten(self, input):
        shape = input.shape
        num_features = shape[1:4].num_elements()
        layer_flat = tf.reshape(input, [1,1,1, num_features])
        print "Flattened layer: "+ str(layer_flat.shape)

        return layer_flat, num_features

    def layer_fc(self, input, num_inputs):
        weights = self.new_weights(shape=[1, 1, num_inputs, self.fc_size])
        biases = self.new_biases(length=self.fc_size)

        layer = tf.matmul(input, weights) + biases

        #layer = self.layer_bn(layer)
        layer = self.layer_relu(layer)
        print "FC layer: "+ str(layer.shape)

        return layer

    def layer_conv_transpose(self, input, kernel_size, num_outputs, strides, padding, output_size, relu, bn):
        input_shape = input.shape
        num_inputs = int(input_shape[3])

        shape = [kernel_size, kernel_size, num_outputs, num_inputs]
        filter = self.new_weights(shape=shape)

        strides = [1, strides, strides, 1]

        out_shape = [int(input_shape[0]), output_size, output_size, num_outputs]
        out = tf.nn.conv2d_transpose(value=input, filter=filter, output_shape=out_shape,
            strides=strides, padding=padding)

        if bn:
            #print "Batch Normal"
            out = self.layer_bn(out)

        if relu:
            #print "ReLu"
            out = self.layer_relu(out)

        return out

    def layer_conv(self, input, num_channels, kernel_size, num_filters, strides, padding, relu, bn):
        shape = [kernel_size, kernel_size, num_channels, num_filters]

        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=num_filters)

        strides = [1, strides, strides, 1]
        conv_op = tf.nn.conv2d(input=input, filter=weights, strides=strides, padding=padding)
        conv_op += biases

        if bn:
            #print "Batch Normal"
            conv_op = self.layer_bn(conv_op)

        if relu:
            #print "ReLu"
            conv_op = self.layer_relu(conv_op)
        #elif relu==False:
        #     conv_op = self.lrelu(1.0, conv_op)

        #print "Convolution: " + str(conv_op.shape)
        return conv_op, weights

    def layer_crop(self, input, scale):
        shape = input.shape
        scale = 1/scale
        target_h = int(shape[1]/scale)
        target_w = int(shape[2]/scale)

        # crop from the center of the image
        offset_h = target_h/2
        offset_w = target_w/2

        out = tf.image.crop_to_bounding_box(image=input, offset_height=offset_h,
            offset_width=offset_w, target_height=target_h, target_width=target_w)

        return out

    def layer_concat(self, layer1, layer2):
        out = layer1 + layer2
        out = tf.nn.relu(out)
        return out

    def layer_conv_res(self, input, num_channels, num_filters):
        out, weights = self.layer_conv(input=input, num_channels=num_channels, kernel_size=3,
            num_filters=num_filters, strides=1, padding='SAME', relu=True, bn=False)
        return out

    def down_res_block(self, input, num_filters):
        print "Down-Residual Block-"+ str(num_filters)

        shape = input.shape
        num_channels = int(shape[3])

        downSampling_res = self.downSampling_res_block(input=input, num_channels=num_channels,
            num_filters=num_filters)

        shape = downSampling_res.shape
        num_channels = int(shape[3])

        res1 = self.res_block(input=downSampling_res, num_channels=num_channels, num_filters=num_filters)

        res2 = self.res_block(input=res1, num_channels=num_channels, num_filters=num_filters)
        return res2

    def downSampling_res_block(self, input, num_channels, num_filters):
        print "\tDown-Sampling Residual block"
        conv1, weights = self.layer_conv(input=input, num_channels=num_channels, kernel_size=1,
            num_filters=num_filters, strides=2, padding='VALID', relu=True, bn=False)
        print "\t\tconv1: "+ str(conv1.shape)

        shape = conv1.shape
        num_channels = int(shape[3])

        conv2 = self.layer_conv_res(input=conv1, num_channels=num_channels, num_filters=num_filters)
        print "\t\tconv2: "+ str(conv2.shape)

        conv3 = self.layer_conv_res(input=conv2, num_channels=num_channels, num_filters=num_filters)
        print "\t\tconv3: "+ str(conv3.shape)

        downSampling_res = self.layer_concat(conv3, conv1)
        print "\t\tdown-sampling residual: "+ str(downSampling_res.shape)

        return downSampling_res

    def res_block(self, input, num_channels, num_filters):
        print "\n\tResidual block"
        conv1 = self.layer_conv_res(input=input, num_channels=num_channels, num_filters=num_filters)
        print "\t\tconv1: "+ str(conv1.shape)

        conv2 = self.layer_conv_res(input=conv1, num_channels=num_channels, num_filters=num_filters)
        print "\t\tconv2: "+ str(conv2.shape)

        res = self.layer_concat(conv2, conv1)
        print "\t\tresidual block: " + str(res.shape)

        return res

    def up_res_block(self, input, num_outputs, output_size):
        print "Up-Residual Block-"+ str(num_outputs)

        upSampling_res = self.upSampling_res_block(input=input, num_outputs=num_outputs, output_size=output_size)

        shape = upSampling_res.shape
        num_channels = int(shape[3])

        res1 = self.res_block(input=upSampling_res, num_channels=num_channels, num_filters=num_outputs)

        res2 = self.res_block(input=res1, num_channels=num_channels, num_filters=num_outputs)

        return res2

    def upSampling_res_block(self, input, num_outputs, output_size):
        print "\tUp-Sampling Residual block"
        conv1 = self.layer_conv_transpose(input=input, kernel_size=2, num_outputs=num_outputs, strides=2,
            padding='SAME', output_size=output_size, relu=True, bn=False)
        print "\t\tconv-transpose1: "+ str(conv1.shape)

        conv2 = self.layer_conv_transpose(input=conv1, kernel_size=2, num_outputs=num_outputs, strides=2,
            padding='SAME', output_size=output_size*2, relu=True, bn=False)
        print "\t\tconv-transpose2: "+ str(conv2.shape)
        conv2 = self.layer_crop(input=conv2, scale=0.5)
        print "\t\tconv2: "+ str(conv2.shape)

        shape = conv2.shape
        num_channels = int(shape[3])

        conv3 = self.layer_conv_res(input=conv2, num_channels=num_channels, num_filters=num_outputs)
        print "\t\tconv3: "+ str(conv3.shape)

        upSampling_res = self.layer_concat(conv3, conv1)
        print "\t\tresidual block: " + str(upSampling_res.shape)

        return upSampling_res

    def write_xyz(self, x, y, z, path, filename):
        #atom = 'C'
        myfile = open(filename+".txt","w")
        #myfile.write("{}\n".format(len(x)))

        for i in range (0, len(x)):
            x_coor, y_coor, z_coor = float (x[i]), float(y[i]), float(z[i])
            myfile.write("{} {} {}\n".format(x_coor,y_coor,z_coor))
            #myfile.write("{}\t {}\t {}\t\n".format(x_coor,y_coor, z_coor))
        myfile.close()


    def write_xy(self, x, y, path, filename):
        #atom = 'C'
        myfile = open(filename+".txt","w")
        #myfile.write("{}\n".format(len(x)))

        for i in range (0, len(x)):
            x_coor, y_coor = float (x[i]), float(y[i])
            myfile.write("{} {}\n".format(x_coor,y_coor))
            #myfile.write("{}\t {}\t {}\t\n".format(x_coor,y_coor, z_coor))
        myfile.close()
