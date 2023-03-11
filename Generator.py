from PIL import Image
import numpy as np
array = np.random.random(size=(416,416,3))
print(array.shape)
array = array*255
array = array.astype(dtype=np.uint8)
print(array)
img = Image.fromarray(array)
img.save("C:/Users/86139/Desktop/1.jpg")