from collections import OrderedDict
import cv2
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot
import pylab

# Load Data from File
def load_file(file_path, filename, num_fields):
    points = np.loadtxt(file_path + filename + '.txt')

    if num_fields==3:
        a,b,c = zip(*points)
        list = np.column_stack((a, b, c))

    elif num_fields==4:
        a,b,c,d = zip(*points)
        list = np.column_stack((a, b, c, d))

    return list

def getAffineTransform(src, dst):
    extra = [1.0, 1.0, 1.0]

    src = np.reshape(src, (2,3))
    dst = np.reshape(dst, (2,3))
    extra = np.reshape(extra, (1,3))

    src = np.vstack((src, extra))

    src_inv = inv(src)

    affineMtx = np.matmul(dst, src_inv)
    return affineMtx

def bbox(points):
    points = np.asarray(points)

    x_coor = points[:,0]
    minx = np.min(x_coor)
    maxx = np.max(x_coor)

    y_coor = points[:,1]
    miny = np.min(y_coor)
    maxy = np.max(y_coor)

    width = maxx-minx
    height = maxy-miny

    return minx, maxx, miny, maxy, width, height

def Barycentric(points):
    points = np.asarray(points)
    # sort by y-axis
    points = points[np.lexsort((points[:,0],points[:,1]))]

    # get the bounding box of the triangle
    minX, maxX, minY, maxY, w, h = bbox(points)

    v21 = [ points[1,0]-points[0,0], points[1,1]-points[0,1] ]
    v31 = [ points[2,0]-points[0,0], points[2,1]-points[0,1] ]
    coords = []

    for x in range(minX, maxX):
        for y in range(minY, maxY):
            q = [x - points[0,0], y - points[0,1]]

            s = np.cross(q, v21) / np.cross(v21, v31)
            t = np.cross(v21, q) / np.cross(v21, v31)

            if ( (s >= 0) and (t >= 0) and (s + t <= 1)):
                # inside triangle
                #print '(',x , ', ', y ,')\t inside triangle'
                coords.append([x,y])
            #else:
                #print '(',x , ', ', y ,')'

    return coords

def getLine(start, end):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def fillTriangle(borders):
    """
        borders: [lineAB, lineAC] --> (n,2)
    """
    borders = np.asarray(borders)
    points=[]

    x,y = borders[0]
    points.append((x,y))

    for i in range(1, len(borders)):
        if (borders[i].tolist() != points[len(points)-1]) :
            #print x,',',y,'\t',borders[i]
            if borders[i, 1] == y :
                for j in range(x+1, borders[i,0]+1):
                    points.append((x+1,y))
                    x = x+1
            else :
                x,y = borders[i]
                points.append((x,y))
    return points


def write_txt(list_, path, filename):
    myfile = open(path+'/'+filename+".txt","w")
    #myfile=open('/home/shrinivasan/Desktop/'+filename+".txt","w")

    # list[i, vertex1, vertex2]
    face_id = list_[:,0]
    vertex1 = list_[:,1]
    vertex2 = list_[:,2]

    for i in range (0, len(list_)):
        vertex_1, vertex_2 = float(vertex1[i]), float(vertex2[i])
        myfile.write("{}\t {}\t {}\t\n".format(face_id[i], vertex_1, vertex_2))
    myfile.close()

def faceRasterization(object_path):
    # Load Template Data
    vertices2 = load_file(file_path=object_path+'/', filename='vertices', num_fields=4)
    edges2 = load_file(file_path=object_path+'/', filename='edges', num_fields=3)
    mesh2 = load_file(file_path=object_path+'/', filename='faces', num_fields=4)

    # Read input image and convert to float
    img1 = cv2.imread(object_path+"/texture.jpg")

    fill = [[0,0,0]]
    for i in range(0,len(mesh2)):
        first = True
        total = 0
        size = 0
        coords_t=[]

        for k in range(1,4):
            edge_t = int(mesh2[i][k])
            #print '\nedge-p: ',edge
            #print 'edge-pt: ',edge_t
            # 3 points
            for l in range(1,3):
                p_t = int(edges2[edge_t][l])
                #print '\npoint-p: ',p
                #print 'point-pt: ',p_t

                size = len(coords_t)
                #print 'size: ', size
                #print 'p: ', vertices[p][1:3]
                #print 'pt: ', vertices2[p_t][1:3]

                if size==0:
                    #print '0: '
                    #print '\tp: ', coords[0]
                    coords_t.append(vertices2[p_t][1:3])

                    #print '\tpt: ', coords_t[0]

                elif size==1:
                    #print '1: '
                    if (coords_t[0][1:3].tolist()) != (vertices2[p_t][1:3].tolist()):
                        coords_t.append(vertices2[p_t][1:3])
                        #print '\tpt: ', coords_t[1]

                elif size==2:
                    #print '2: '
                        #print '\tp: ', coords[2]
                    if (coords_t[0].tolist()) != (vertices2[p_t][1:3].tolist()) and \
                        (coords_t[1].tolist()) != (vertices2[p_t][1:3].tolist()):
                        coords_t.append(vertices2[p_t][1:3])
                        #print '\tpt: ', coords_t[2]

                elif len(coords_t)>2:
                    break

        # Define input and output triangles
        tri2 = np.asarray(coords_t)
        tri2 = tri2.astype(np.int32)

        #print '\nFaces: ', i
        #print 'tri1: \n', tri1
        #print 'tri2: \n',tri2

        point = tri2[np.lexsort((tri2[:,0],tri2[:,1]))]
        #print 'point\n',point

        line12 = getLine(point[0], point[1])
        line13 = getLine(point[0], point[2])
        line23 = getLine(point[1], point[2])

        borders = np.vstack((line12, line13, line23))
        borders = borders[np.lexsort((borders[:,0],borders[:,1]))]

        tri_points = fillTriangle(borders)
        tri_points = np.asarray(tri_points)
        #print 'points-',i,'\t',tri_points.shape

        x = np.reshape(tri_points[:,0], (-1,1))
        y = np.reshape(tri_points[:,1], (-1,1))
        one = np.ones_like(x)

        temp = np.hstack((one*i, x, y))
        fill = np.vstack((fill, temp))

    #print np.asarray(fill).shape
    #print np.max(fill)

    fill = fill[1:]

    # remove duplicate points
    tmp = OrderedDict()
    for point in zip(fill[:,0], fill[:,1], fill[:,2]):
        tmp.setdefault(point[:2], point)

    mypoints = tmp.values()
    mypoints = np.asarray(mypoints)

    # Save path
    filename = 'face_filling'
    write_txt(fill, object_path, filename)

    #---------------------------------------
    # Normalize Coordinates
    x = np.asarray(fill[:,1])
    y = np.asarray(fill[:,2])

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    delta_x = max(x)-min(x)
    x_norm = (x-min(x))/delta_x

    delta_y = max(y)-min(y)
    y_norm = (y-min(y))/delta_y

    #x_norm = (x_norm*2)-1
    #y_norm = (y_norm*2)-1

    xy = np.stack([np.asarray(fill[:,0]), x_norm, y_norm], 1)
    write_txt(xy, object_path, filename+'_norm')
    #---------------------------------------

    x = np.reshape(fill[:,1], (1,-1))
    y = np.reshape(fill[:,2], (1,-1))
    matplotlib.pyplot.scatter(x,y)
    matplotlib.pyplot.show()
