#!/usr/bin/env python
# coding=utf-8

import os.path as osp
import sys
sys.path.insert(0, '..')
import mtcnn as mt
import cv2 as cv
from IPython import embed
from pprint import pprint

mtcnn = mt.Mtcnn()
print(mtcnn)

image = osp.join(osp.dirname(__file__), '1.jpg')
im = cv.imread(image)
infos = mtcnn.detect(im)
print('Detect infos of %s' % image)
pprint(infos)
im1 = mtcnn.draw(im, infos)
cv.imwrite('detect.png', im1)

infos = mtcnn.detect_again(im, infos[0].bbox)
print('Detect again infos of %s' % image)
pprint(infos)
im2 = mtcnn.draw(im, infos)
cv.imwrite('detect_again.png', im2)

print('test done.')

