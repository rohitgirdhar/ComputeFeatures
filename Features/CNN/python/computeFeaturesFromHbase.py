import sys, os
caffe_root = '../external/caffe/'
sys.path.insert(0, caffe_root + '/python')
import caffe
sys.path.insert(0, '../external/DiskVector/python')
import PyDiskVectorLMDB
import happybase
import base64
import skimage.io
from StringIO import StringIO
import numpy as np

def extractFeatures():
  pass

def convertJPEGb64ToCaffeImage(img_data_coded):
  img_data = base64.b64decode(img_data_coded)
  # inspired from caffe.io.load_image
  img = skimage.io.imread(StringIO(img_data))
  img = skimage.img_as_float(img).astype(np.float32)
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    if color:
      img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
      img = img[:, :, :3]
  return img

def loadCaffeModels():
  MODEL_FILE = os.path.join('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/run_scripts/deploy.prototxt')
  PRETRAINED = os.path.join('/home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
  mean_image = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
  print np.shape(mean_image)
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
      mean=mean_image,
      channel_swap=(2,1,0), raw_scale=255,
      image_dims=(256,256))
  return net

def readList(fpath):
  f = open(fpath)
  res = f.read().splitlines()
  f.close()
  return res

def main():
  conn = happybase.Connection('10.1.94.57')
  tab = conn.table('roxyscrape')
  model = loadCaffeModels()
  imgslist = readList('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/lists/Images.txt')

  I = convertJPEGb64ToCaffeImage(tab.row('19186517')['image:orig'])

if __name__ == '__main__':
  main()


