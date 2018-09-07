from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf

class ImageElements(object):

    def __init__(self, process_type, points, image, path, object_name, row, col):
        self.row = row #7
        self.col = col #31
        self.image = image

        if process_type == 'template':
            self.object = object_name
            self.path = path + self.object

            #self.scale = self.scale_vector()
            self.vertices = self.vertices()
            self.edges()
            self.faces()

            #self.edges = tf.convert_to_tensor(self.edges)

            self.show_output()

        elif process_type == 'non-template':
            """
            vertices = []
            step = 4096/((self.row+1)*(self.col+1))

            for i in range (0, tf.size(points), step):
                vertices.append(tf.slice(points, [i,0], [1,2]))
                #vertices.append(points[i])

            idx=[]
            for i in range(0, len(vertices)):
                idx.append(i)

            self.vertices = np.asarray(vertices)
            self.vertices[:,0] = idx
            self.write_txt(list=self.vertices, filename='vertices')
            print len(self.vertices)
            """
            self.vertices = points
            self.edges(save=False, typename='non-template')
            self.faces(save=False, typename='non-template')

            #self.show_output()

    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        return self.edges

    def get_faces(self):
        return self.faces

    def get_scale(self):
        return self.scale

    def write_txt(self, list, filename):
        myfile = open(self.path+'/'+filename+".txt","w")
        #myfile=open('/home/anneke/Desktop/'+filename+".txt","w")

        if filename=='vertices':
            # list[i, x, y, z]
            idx = list[:,0]
            x = list[:,1]
            y = list[:,2]
            z = list[:,3]

            delta_x = max(x)-min(x)
            x_norm = (x-min(x))/delta_x

            delta_y = max(y)-min(y)
            y_norm = (y-min(y))/delta_y

            #x_norm = (x_norm*2)-1
            #y_norm = (y_norm*2)-1

            delta_z = max(z)-min(z)
            if delta_z == 0 :
                z_norm = z
            else:
                #z_norm = (z-min(z))/delta_z
                z_norm = (z_norm*2)-1

            for i in range (0, len(list)):
                x_coor, y_coor, z_coor = str(x[i]), str(y[i]), str(z[i])
                myfile.write("{}\t {}\t {}\t {}\t\n".format(idx[i], x_coor, y_coor, z_coor))

            # Normalize Coordinates
            myfile = open(self.path+'/'+filename+"_norm.txt","w")
            for i in range (0, len(list)):
                x_coor, y_coor, z_coor = str(x_norm[i]), str(y_norm[i]), str(z_norm[i])
                myfile.write("{}\t {}\t {}\t {}\t\n".format(idx[i], x_coor, y_coor, z_coor))
                #myfile.write("{}\t {}\t {}\t\n".format(x_coor, y_coor, z_coor))


        elif filename=='edges':
            # list[i, vertex1, vertex2]
            idx = list[:,0]
            vertex1 = list[:,1]
            vertex2 = list[:,2]

            for i in range (0, len(list)):
                vertex_1, vertex_2 = int(vertex1[i]), int(vertex2[i])
                myfile.write("{}\t {}\t {}\t\n".format(idx[i], vertex_1, vertex_2))

        elif filename=='faces':
            # list[i, v=edge1, edge2, edge3]
            idx = list[:,0]
            edge1 = list[:,1]
            edge2 = list[:,2]
            edge3 = list[:,3]

            for i in range (0, len(list)):
                edge_1, edge_2, edge_3 = int(edge1[i]), int(edge2[i]), int(edge3[i])
                myfile.write("{}\t {}\t {}\t {}\t\n".format(idx[i], edge_1, edge_2, edge_3))

        elif filename=='scale_vector':
            myfile.write("{}\t\n{}\t".format(list[0], list[1]))
        myfile.close()

    def scale_vector(self):
        if self.object == 'forest':
            real_width = 20.3
            real_height = 14.2
        img_size = self.image.shape
        #print img_size
        real_size = [real_height, real_width]
        real_size = np.matrix(real_size)
        im_size = [img_size[0], img_size[1]]
        im_size = np.matrix(im_size)
        scale = real_size.I * im_size

        self.write_txt(list=scale, filename='scale_vector')

        return scale

    def faces(self, save=True, typename=None):
        # Triangular Mesh
        row = self.row
        col = self.col

        if typename == 'non-template':
            self.idx_edges = tf.slice(self.edges, [0,0], [-1,1])

            total_faces = row*col*2
            total_edges = tf.size(self.idx_edges)

            # col = idx, edge1, edge2, edge3
            """
            row_faces, col_faces = total_faces, 4
            faces = [[0 for x in range(col_faces)] for y in range(row_faces)]
            faces = np.asarray(faces)
            """

            self.faces = []
            """
            ____
            \   |
             \  |
              \ |
               \|

               vertex1, vertex2, vertex3 = horizontal, vertical, diagonal

            """
            start_vertical = (row*col) + row + col
            start_diagonal = total_edges - (row*col)
            idx_face = 0
            for i in range(0,row):
                for j in range(0,col):
                    idx_face2 = tf.cond(( (start_vertical + (row*j)) < total_edges), \
                        lambda: self.face_1(idx_face, start_vertical, row, j, \
                        start_diagonal, total_edges), lambda: idx_face)

                    if (idx_face2 != idx_face):
                        start_diagonal = start_diagonal+1

                    idx_face = idx_face2

                start_vertical = start_vertical + 1

            """

            |\
            | \
            |  \
            -----

            """
            start_horizontal = col
            start_diagonal = total_edges - (row*col)
            start_vertical = col * (row+1)
            for i in range(0,row):
                for j in range(0,col):
                    idx_face2 = 0
                    idx_face2 = tf.cond(( (start_vertical + (row*j)) < total_edges), \
                        lambda: self.face_extra2(idx_face, start_horizontal, start_vertical, \
                        row, j, start_diagonal), lambda: idx_face)

                    if (idx_face2 != idx_face):
                        start_diagonal = start_diagonal+1
                        start_horizontal = start_horizontal+1

                    idx_face = idx_face2


                start_vertical = start_vertical + 1
        else :
            idx_edges = self.edges[:,0]

            total_faces = row*col*2
            total_edges = len(idx_edges)

            # col = idx, edge1, edge2, edge3
            row_faces, col_faces = total_faces, 4
            faces = [[0 for x in range(col_faces)] for y in range(row_faces)]
            faces = np.asarray(faces)

            """
            ____
            \   |
             \  |
              \ |
               \|

               vertex1, vertex2, vertex3 = horizontal, vertical, diagonal

            """
            start_vertical = (row*col) + row + col
            start_diagonal = len(idx_edges) - (row*col)
            idx_face = 0
            for i in range(0,row):
                for j in range(0,col):
                    if ((start_vertical + (row*j)) < total_edges) and (start_diagonal < total_edges) :
                        faces[idx_face][0] = idx_edges[idx_face]
                        faces[idx_face][1] = idx_edges[idx_face]
                        faces[idx_face][2] = idx_edges[start_vertical + (row*j)]
                        faces[idx_face][3] = idx_edges[start_diagonal]

                        #print 'first: \t',idx_face,'\t',idx_face, '\t', (start_vertical + (row*j)), '\t', start_diagonal
                        start_diagonal = start_diagonal + 1
                        idx_face = idx_face + 1
                start_vertical = start_vertical + 1

            """

            |\
            | \
            |  \
            -----

            """
            start_horizontal = col
            start_diagonal = len(idx_edges) - (row*col)
            start_vertical = col * (row+1)
            for i in range(0,row):
                for j in range(0,col):
                    if (start_vertical + (row*j)) < total_edges :
                        faces[idx_face][0] = idx_edges[idx_face]
                        faces[idx_face][1] = idx_edges[start_horizontal]
                        faces[idx_face][2] = idx_edges[start_vertical + (row*j)]
                        faces[idx_face][3] = idx_edges[start_diagonal]

                        #print 'secon: \t',idx_face,'\t',start_horizontal, '\t', (start_vertical + (row*j)), '\t', start_diagonal
                        start_diagonal = start_diagonal + 1
                        start_horizontal = start_horizontal + 1
                        idx_face = idx_face + 1
                start_vertical = start_vertical + 1

            self.faces = faces
        if save:
            self.write_txt(list=self.faces, filename='faces')


    def face_1(self, idx_face, start_vertical, row, j, start_diagonal, total_edges):
        idx_face = tf.cond((start_diagonal < total_edges), \
            lambda: self.face_extra1(idx_face, start_vertical, row, j, start_diagonal), lambda: idx_face)

        return idx_face

    def face_extra1(self, idx_face, start_vertical, row, j, start_diagonal):
        slice1 = tf.slice(self.idx_edges, [idx_face, 0], [1,1])
        slice2 = tf.slice(self.idx_edges, [idx_face, 0], [1,1])
        slice3 = tf.slice(self.idx_edges, [start_vertical + (row*j), 0], [1,1])
        slice4 = tf.slice(self.idx_edges, [start_diagonal, 0], [1,1])
        self.faces.append([slice1, slice2, slice3, slice4])
        """
        faces[idx_face][0] = idx_edges[idx_face]
        faces[idx_face][1] = idx_edges[idx_face]
        faces[idx_face][2] = idx_edges[start_vertical + (row*j)]
        faces[idx_face][3] = idx_edges[start_diagonal]

        #print 'first: \t',idx_face,'\t',idx_face, '\t', (start_vertical + (row*j)), '\t', start_diagonal
        """

        idx_face = idx_face + 1
        return idx_face

    def face_extra2(self, idx_face, start_horizontal, start_vertical, row, j, start_diagonal):
        slice1 = tf.slice(self.idx_edges, [idx_face, 0], [1,1])
        slice2 = tf.slice(self.idx_edges, [start_horizontal, 0], [1,1])
        slice3 = tf.slice(self.idx_edges, [start_vertical + (row*j), 0], [1,1])
        slice4 = tf.slice(self.idx_edges, [start_diagonal, 0], [1,1])

        self.faces.append([slice1, slice2, slice3, slice4])

        """
        faces[idx_face][0] = idx_edges[idx_face]
        faces[idx_face][1] = idx_edges[start_horizontal]
        faces[idx_face][2] = idx_edges[start_vertical + (row*j)]
        faces[idx_face][3] = idx_edges[start_diagonal]
        """

        idx_face = idx_face + 1
        return idx_face

    def edges(self, save=True, typename=None):
        col = self.col
        row = self.row

        #print self.vertices.shape

        if typename == 'non-template':
            self.idx_points = tf.slice(self.vertices, [0,0], [-1,1])
            total_points = tf.size(self.idx_points)

            total_edges = (row*(col+1)) + (col*(row+1)) + (row*col)

            # col = idx, vertex1, vertex2

            """
            row_edges, col_edges = total_edges, 3
            edges = [[0 for x in range(col_edges)] for y in range(row_edges)]
            self.edges = np.asarray(edges)
            """

            #self.edges = []

            idx_edge = 0
            idx_pts = 0

            # horizontal edges
            for i in range(0, row+1):
                for j in range(0, col):
                    idx_edge = tf.cond( (idx_pts+j+1) < total_points, \
                        lambda: self.edge_1(idx_edge, idx_pts, j, total_points), lambda: idx_edge)
                idx_pts = idx_pts + col + 1

            # vertical edges
            step = col + 1
            for i in range(0, col+1):
                for j in range(0, row):
                    idx_edge = tf.cond(((step*j)+i) < total_points, \
                        lambda: self.edge_2(idx_edge, step, j, i, total_points), lambda: idx_edge)
            # diagonal edges
            step = col + 2
            start = 0
            for i in range(0, row):
                for j in range(0, col):
                    idx_edge = tf.cond((j+start) < total_points, \
                        lambda: self.edge_3(idx_edge, step, start, j, total_points), lambda: idx_edge)
                start = start + col + 1

            #self.edges2 = tf.convert_to_tensor(self.edges)
        else :
            idx_points = self.vertices[:,0]
            total_points = len(idx_points)

            total_edges = (row*(col+1)) + (col*(row+1)) + (row*col)

            # col = idx, vertex1, vertex2
            row_edges, col_edges = total_edges, 3
            edges = [[0 for x in range(col_edges)] for y in range(row_edges)]
            edges = np.asarray(edges)

            idx_edge = 0
            idx_pts = 0

            # horizontal edges
            for i in range(0, row+1):
                for j in range(0, col):
                    if (idx_pts+j+1) < (total_points) and (idx_pts+j) < (total_points):
                        edges[idx_edge][0]=idx_edge
                        edges[idx_edge][1]=idx_points[idx_pts + j]
                        edges[idx_edge][2]=idx_points[idx_pts + j + 1]
                        #print 'hor: \t',idx_edge,'\t',(idx_pts+j), '\t', (idx_pts+j+1)
                        idx_edge = idx_edge + 1
                idx_pts = idx_pts + col + 1

            # vertical edges
            step = col + 1
            for i in range(0, col+1):
                for j in range(0, row):
                    if ((step*j)+i) < (total_points) and ((step*(j+1))+i) < (total_points) :
                        edges[idx_edge][0]=idx_edge
                        edges[idx_edge][1]=idx_points[(step*j)+i]
                        edges[idx_edge][2]=idx_points[(step*(j+1))+i]
                        #print 'ver: \t',idx_edge,'\t',((step*j)+i), '\t', ((step*(j+1))+i)
                        idx_edge = idx_edge + 1

            # diagonal edges
            step = col + 2
            start = 0
            for i in range(0, row):
                for j in range(0, col):
                    if (j+start) < (total_points) and (step + j + start) < (total_points) :
                        edges[idx_edge][0]=idx_edge
                        edges[idx_edge][1]=idx_points[j+start]
                        edges[idx_edge][2]=idx_points[step + j + start]
                        #print 'dia: \t',idx_edge,'\t',(j+start), '\t', (step + j + start)
                        idx_edge = idx_edge + 1
                start = start + col + 1

            self.edges = edges

        if save:
            self.write_txt(list=self.edges, filename='edges')

    def edge_1(self, idx_edge, idx_pts, j, total_points):
        idx_edge = tf.cond(( (idx_pts+j) < total_points), \
            lambda: self.edge_extra1(idx_edge, idx_pts, j), lambda: idx_edge)

        return idx_edge

    def edge_2(self, idx_edge, step, j, i, total_points):
        idx_edge = tf.cond(( ((step*(j+1))+i) < total_points), \
            lambda: self.edge_extra2(idx_edge, step, j, i), lambda: idx_edge)

        return idx_edge

    def edge_3(self, idx_edge, step, start, j, total_points):
        idx_edge = tf.cond(( (step + j + start) < total_points), \
            lambda: self.edge_extra3(idx_edge, step, start, j), lambda: idx_edge)

        return idx_edge

    def edge_extra1(self, idx_edge, idx_pts, j):
        slice_ = tf.slice(self.idx_points, [idx_pts + j, 0], [1,1])
        slice__ = tf.slice(self.idx_points, [idx_pts + j + 1, 0], [1,1])

        idx = tf.reshape(idx_edge, [1,1])
        #self.edges.append([idx, slice_, slice__])

        if idx_edge == 0 :
            self.edges = tf.concat([idx, slice_, slice__], 1)
            #print 'edgess: ', self.edges.shape
        else :
            temp = tf.concat([idx, slice_, slice__], 1)
            self.edges = tf.concat([self.edges, temp], 0)

        """
        self.edges[idx_edge][0]=idx_edge
        self.edges[idx_edge][1]=self.idx_points[idx_pts + j]
        self.edges[idx_edge][2]=self.idx_points[idx_pts + j + 1]
        """
        #print 'hor: \t',idx_edge,'\t',(idx_pts+j), '\t', (idx_pts+j+1)
        idx_edge = idx_edge+1
        return idx_edge

    def edge_extra2(self, idx_edge, step, j, i):
        slice_ = tf.slice(self.idx_points, [(step*j)+i, 0], [1,1])
        slice__ = tf.slice(self.idx_points, [(step*(j+1))+i, 0], [1,1])
        idx = tf.reshape(idx_edge, [1,1])
        #self.edges.append([idx, slice_, slice__])

        temp = tf.concat([idx, slice_, slice__], 1)
        self.edges = tf.concat([self.edges, temp], 0)

        """
        self.edges[idx_edge][0]=idx_edge
        self.edges[idx_edge][1]=self.idx_points[(step*j)+i]
        self.edges[idx_edge][2]=self.idx_points[(step*(j+1))+i]
        """
        #print 'ver: \t',idx_edge,'\t',((step*j)+i), '\t', ((step*(j+1))+i)
        idx_edge = idx_edge+1
        return idx_edge

    def edge_extra3(self, idx_edge, step, start, j):
        slice_ = tf.slice(self.idx_points, [j+start, 0], [1,1])
        slice__ = tf.slice(self.idx_points, [step + j + start, 0], [1,1])
        idx = tf.reshape(idx_edge, [1,1])
        #self.edges.append([idx, slice_, slice__])

        temp = tf.concat([idx, slice_, slice__], 1)
        self.edges = tf.concat([self.edges, temp], 0)

        """
        self.edges[idx_edge][0]=idx_edge
        self.edges[idx_edge][1]=self.idx_points[j+start]
        self.edges[idx_edge][2]=self.idx_points[step + j + start]
        """
        #print 'dia: \t',idx_edge,'\t',(j+start), '\t', (step + j + start)
        idx_edge = idx_edge+1
        return idx_edge

    def vertices(self):
        img_size = self.image.shape
        #height, width, dim = img_size
        self.height, self.width = img_size

        total_points = (self.row+1)*(self.col+1)

        i = np.zeros(total_points)
        x = np.zeros(total_points)
        y = np.zeros(total_points)

        #idx = 0
        #start_x = start_y = 0

        #step_x = float(width)/float(self.col)
        #step_y = float(height)/float(self.row)
        #print step_x, '\t', step_y

        # col = idx, x, y, z
        row, col = total_points, 4
        vertices = [[0 for x in range(col)] for y in range(row)]
        vertices = np.asarray(vertices)
        vertices = vertices.astype(np.float32)
        #print vertices.shape

        x = np.linspace(0, self.width-1, num=self.col+1)
        y = np.linspace(0, self.height-1, num=self.row+1)
        x = np.floor(x)
        y = np.floor(y)

        idx=0

        for i in range(0, len(y)):
            for j in range(0, len(x)):
                vertices[idx,0] = idx
                vertices[idx,1] = x[j]
                vertices[idx,2] = y[i]

                idx = idx+1

        vertices[:,3] = 0.
        """
        while (idx < total_points):

            vertices[idx][1] = int(math.floor(start_x))
            vertices[idx][2] = int(math.floor(start_y))
            vertices[idx][3] = 0

            #print '\n1: ', vertices[idx][1], ' ', start_x
            #print int(math.ceil(start_x)) , '\t', math.floor(start_x), '\t', start_x
            if ( ((int(math.ceil(start_x)) % width) == 0) or ((int(math.ceil(start_x)) % width) == 1) )\
                and (start_x != 0):
                #print 'y: ', start_y, '\t', step_y
                start_y = start_y + step_y
                #print start_y
                start_x = 0
            else:
                #print 'x: ',start_x, '\t', step_x
                start_x = start_x + step_x
                #print start_x

            #print 'vert: \t',idx,'\t',x[idx], '\t', y[idx]

            vertices[idx][0] = idx
            idx = idx + 1
        """
        temp = np.zeros_like(vertices)
        temp[:,1] = vertices[:,1] / (self.width-1)
        temp[:,2] = vertices[:,2] / (self.height-1)
        self.write_txt(list=vertices, filename='vertices')

        return vertices

    def show_output(self):
        # show
        i = Image.open('/home/anneke/project/surf/tensorflow/data/objects/forest/texture.jpg')
        #implot = plt.imshow(i)
        implot = plt.imshow(self.image, cmap='gray')

        # Show points
        print self.vertices.shape
        plt.scatter(x=self.vertices[:,1], y=self.vertices[:,2], c='r', s=10)

        # Show mesh
        for i in range (0,len(self.edges)):
            x=np.zeros(2)
            y=np.zeros(2)
            vertex1 = self.edges[i][1]
            vertex2 = self.edges[i][2]
            #print '\nvertex: ',vertex1,'\t', vertex2

            x[0] = self.vertices[vertex1][1]
            y[0] = self.vertices[vertex1][2]
            x[1] = self.vertices[vertex2][1]
            y[1] = self.vertices[vertex2][2]

            plt.plot(x,y, 'ro-')
            #print 'p1: ', vertex1, '\t', vertices[vertex1][1], '\t', vertices[vertex1][2]
            #print 'p2: ', vertex2, '\t', vertices[vertex2][1], '\t', vertices[vertex2][2]

        plt.show()
