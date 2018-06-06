# MTCNN 
- python extension with c++ implementation

## 1. Install
```shell
git clone https://github.com/yangfly/mtcnn.git
cd mtcnn
# set caffe_root to your local caffe in setup.py
make -j 4
```

## 2. Test
```shell
make test
```

### 3. Todo
- [x] Debug
 - BBox initalization: [code](https://github.com/yangfly/mtcnn/blob/master/src/core/pybing.cpp#L386)
 - Reference: [Blog](https://blog.csdn.net/jxlczjp77/article/details/1783456)

