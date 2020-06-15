from PIL import Image
import numpy as  np
import matplotlib.pyplot as  plt
import matplotlib

im = matplotlib.image.imread('../B/聚类/01.jpg')
print(im.shape)

l,w,s =tuple(im.shape)
print(l,w,s)

