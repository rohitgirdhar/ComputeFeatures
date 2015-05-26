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
import time

def convertJPEGb64ToCaffeImage(img_data_coded):
  try:
    img_data = base64.b64decode(img_data_coded)
    img = skimage.io.imread(StringIO(img_data))
  except:
    return np.zeros((256,256,3))
  # inspired from caffe.io.load_image
  img = skimage.img_as_float(img).astype(np.float32)
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    if color:
      img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
      img = img[:, :, :3]
  return img

def loadCaffeModels():
  MODEL_FILE = os.path.join('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/run_scripts/deploy_pool5.prototxt')
  PRETRAINED = os.path.join('/home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
  mean_image = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
  print np.shape(mean_image)
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
      mean=mean_image,
      channel_swap=(2,1,0), raw_scale=255,
      image_dims=(256,256))
  return net

# @return: a nImgs x 9216D numpy array
def extractPool5Features(imgs, model, normalize = False):
  features = model.predict(imgs)
  if normalize:
    row_norms = np.linalg.norm(features, 2, axis=1)
    features = features / row_norms[:, np.newaxis]
  return features

def readList(fpath):
  f = open(fpath)
  res = f.read().splitlines()
  f.close()
  return res

def getImagesFromIds(ids, hbasetable):
  imgs = []
  for i in ids:
    imgs.append(convertJPEGb64ToCaffeImage(hbasetable.row(i)['image:orig']))
  return imgs

def saveFeat(feat, id, stor):
  f = PyDiskVectorLMDB.FeatureVector()
  for i in range(np.shape(feat)[0]):
    f.append(float(feat[i]))
  stor.Put(id, f)

def runFeatExt(imgslist, model, hbasetable, stor, normalize = False):
  batchSize = model.blobs['data'].num
  cur_pos = 0
  while cur_pos < len(imgslist):
    print('Doing for %s (%d / %d)' %(imgslist[cur_pos], cur_pos, len(imgslist)))
    start_time = time.time()
    next_pos = min(cur_pos + batchSize, len(imgslist))
    batch = imgslist[cur_pos : next_pos]
    imgs = getImagesFromIds(batch, hbasetable)
    loadImg_time = time.time()
    feats = extractPool5Features(imgs, model, normalize)
    featExt_time = time.time()
    # save the feats
    j = 0
    for i in range(cur_pos, next_pos):
      saveFeat(feats[j, :], i * 10000 + 1, stor)
      j += 1
    save_time = time.time()
    print('Done in \n\tTotal: %d msec\n\tLoad: %d\n\tFeatExt: %d\n\tSave: %d' 
        % ((save_time - start_time) * 1000, 
           (loadImg_time - start_time) * 1000, 
           (featExt_time - loadImg_time) * 1000, 
           (save_time - featExt_time) * 1000))
    cur_pos = next_pos

def main():
  caffe.set_mode_gpu()
  conn = happybase.Connection('10.1.94.57')
  tab = conn.table('roxyscrape')
  model = loadCaffeModels()
  stor = PyDiskVectorLMDB.DiskVectorLMDB('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/Features/pool5_normed', False)
  imgslist = readList('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/lists/Images.txt')
  runFeatExt(imgslist, model, tab, stor, normalize=True)

if __name__ == '__main__':
  main()


