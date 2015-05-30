import sys, os
import happybase
import base64
from StringIO import StringIO
import numpy as np
import time
import operator
import sha
import time

last_print = time.time()
def tic_toc_print(msg):
  global last_print
  if time.time() > last_print + 1:
    print(msg)
    last_print = time.time()

def readList(fpath):
  f = open(fpath)
  res = f.read().splitlines()
  f.close()
  return res

def saveFeat(feat, fpath):
  f = open(fpath, 'a')
  f.write(feat + '\n')
  f.close()

def extractSha1Hash(img):
  obj = sha.new(img)
  return obj.hexdigest()

def readImage(imid, tab):
  return tab.row(imid)['image:orig']

def runFeatExt(imgslist, hbasetable, start_pos):
  cur_pos = start_pos
  while cur_pos < len(imgslist):
    tic_toc_print('Doing for %s (%d / %d)' %(imgslist[cur_pos], cur_pos, len(imgslist)))
    img = readImage(imgslist[cur_pos], hbasetable)
    feats = extractSha1Hash(img)
    saveFeat(feats, '/home/rgirdhar/memexdata/Dataset/processed/0004_IST/Features/SHA1/hashes.txt')
    cur_pos += 1

def main():
  conn = happybase.Connection('10.1.94.57')
  tab = conn.table('roxyscrape')
  imgslist = readList('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/lists/Images.txt')
  start_pos = 0 # default = 0
  runFeatExt(imgslist, tab, start_pos)

if __name__ == '__main__':
  main()


