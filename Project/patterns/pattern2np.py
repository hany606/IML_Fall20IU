from PIL import Image
import numpy as np

im = Image.open("pat1.png")
im_np = np.asarray(im).reshape((28,28,1))
print(im_np.shape)
print(im_np)
