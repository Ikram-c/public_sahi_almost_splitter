import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(0, 2, size=(1920, 1080))
plt.imshow(img)
plt.show()

print("done")


np.hsplit(img, 2)