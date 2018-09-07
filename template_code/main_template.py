import argparse
import cv2
from image_elements import *
from FaceRasterization import *

parser = argparse.ArgumentParser(description='Template Attributes Generator')
parser.add_argument('--object',     type=str, help='object folder name', default='forest')
parser.add_argument('--path',     type=str, help='path to the object home', default='/home/anneke/project/surf/tensorflow/data/objects/')
parser.add_argument('--row',     type=int, help='the total row for template grid', default='32')
parser.add_argument('--col',     type=int, help='the total col for template grid', default='32')

args = parser.parse_args()

texture = cv2.imread(args.path + args.object + '/texture.jpg')
texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)

row = args.row - 1
col = args.col - 1

# Generate vertices, edges, and mesh
ImageElements(process_type='template', points=None, \
    image=texture, path=args.path, object_name=args.object, row=row, col=col)

# Face Filled Points
faceRasterization(object_path=args.path+args.object)
