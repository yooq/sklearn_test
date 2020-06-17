import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import matplotlib

# china = load_sample_image('china.jpg')


china = matplotlib.image.imread('01.jpg')
l,w,c = tuple(china.shape)

plt.figure()
plt.imshow(china)
plt.title('1')
plt.show()

china =np.array(china)/china.max()

china =china.reshape((l*w,c))

print(china.shape)

imagearr = shuffle(china,random_state=0)[0:1000]

kmodel = KMeans(n_clusters=64,random_state=0)
kmodel = kmodel.fit(imagearr)
center = kmodel.cluster_centers_
print(center)

pre = kmodel.predict(china)
print(pre)

imag = np.array([center[i] for i in pre])
print(imag.shape)
imag = imag.reshape(l,w,c)
print(imag.shape)

plt.imshow(imag)
plt.title('2')
plt.show()




