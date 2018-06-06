# coding=utf-8
import _mtcnn as mt
import os
from easydict import EasyDict as edict

### Log Wrapper with side effects
class logger(object):
    level = -1

    @staticmethod
    def open(level):
        if level < 0:
            return
        logger.close()
        logger.level = level
        mt.init_log(level)

    @staticmethod
    def close():
        if logger.level >= 0:
            logger.level = -1
            mt.close_log()

    @staticmethod
    def log(info):
        if logger.level >= 0:
            mt.log(info)

## MTcnn wrapper
class Mtcnn(object):
    def __init__(self, model_dir='', precise_landmark=True, gpu_id=0):
        # temp close log info
        mt.set_device(gpu_id)
        logger.open(2)
        if model_dir == '':
            model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src/models')
        if not os.path.exists(os.path.join(model_dir, 'MTCNN_det4.prototxt')):
            precise_landmark = False
        self.mtcnn = mt.MTcnn(model_dir, precise_landmark)
        # reopen log info
        logger.open(0)

    @property
    def factor(self):
        return self.mtcnn.factor
    @factor.setter
    def factor(self, value):
        self.mtcnn.factor = value

    @property
    def min_size(self):
        return self.mtcnn.minSize
    @min_size.setter
    def min_size(self, value):
        self.mtcnn.minSize = value

    @property
    def max_size(self):
        return self.mtcnn.maxSize
    @max_size.setter
    def max_size(self, value):
        self.mtcnn.maxSize = value

    @property
    def thresholds(self):
        return self.mtcnn.thresholds
    @thresholds.setter
    def thresholds(self, value):
        self.mtcnn.thresholds = value

    @property
    def precise_landmark(self):
        return self.mtcnn.preciseLandmark
    @precise_landmark.setter
    def precise_landmark(self, value):
        self.mtcnn.preciseLandmark = value

    def _decode_infos(self, raw_infos):
        infos = []
        for i in range(0, len(raw_infos), 15):
            raw_info = raw_infos[i:(i+15)]
            info = edict()
            info.score = raw_info[0]
            info.bbox = raw_info[1:5]
            info.fpts = raw_info[5:]
            infos.append(info)
        return infos

    def _encode_infos(self, infos):
        raw_infos = []
        for info in infos:
            raw_infos += [info.score] + info.bbox + info.fpts
        return raw_infos
    
    def detect(self, im, precise_landmark=True):
        raw_infos = self.mtcnn.detect(im, precise_landmark)
        return self._decode_infos(raw_infos)
    
    def detect_again(self, im, bbox, precise_landmark=True):
        bbox = map(float, bbox)
        raw_infos = self.mtcnn.detect_again(im, bbox, precise_landmark)
        return self._decode_infos(raw_infos)

    def align(self, im, fpts, width=112):
        return self.mtcnn.align(im, fpts, width)

    def draw(self, im, infos):
        raw_infos = self._encode_infos(infos)
        return self.mtcnn.draw_infos(im, raw_infos)

    def __str__(self):
        pclm = 'True' if self.precise_landmark else 'False'
        str_ = ' MTCNN settings:\n' + \
               '   mtcnn.min_size = %d\n' % self.min_size + \
               '   mtcnn.max_size = %d\n' % self.max_size + \
               '   mtcnn.factor = %.3f\n' % self.factor + \
               '   mtcnn.precise_landmark = ' + '%s\n' % pclm + \
               '   mtcnn.thresholds = [%.1f, %.1f, %.1f]' % tuple(self.thresholds)
        return str_

