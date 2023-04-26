
import pylab as plt

import numpy as np

'''
fig = plt.figure()
ax0 = fig.add_subplot(131)
image = np.random.poisson(10., (100, 80))
ax0.imshow(image, interpolation='nearest')
ax1 = fig.add_subplot(132)
image = np.random.poisson(10., (100, 80))
i = ax1.imshow(image, interpolation='nearest')
ax1 = fig.add_subplot(133)
fig.colorbar(i, cax = ax1) 


plt.show()
'''
'''
#fig = plt.figure()
fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)
outer_grid = fig11.add_gridspec(1, 2, wspace=10.0, hspace=0.0)
image = np.random.poisson(10., (100, 80))
ax0 = plt.subplot(121)
plt.imshow(image, interpolation='nearest')
image = np.random.poisson(10., (100, 80))
ax1 = plt.subplot(122)
i = plt.imshow(image, interpolation='nearest')
#ax1 = fig.add_subplot(133)
cbar_ax = plt.gcf().add_axes([0.95, 0.15, 0.05, 0.7])
plt.colorbar(i, cax=cbar_ax)

plt.show()


import pylab as plt
import numpy as np
my_image1 = np.linspace(0, 10, 10000).reshape(100,100)
my_image2 = np.sqrt(my_image1.T) + 3
plt.subplot(121)
plt.imshow(my_image1, vmin=0, vmax=10, cmap='jet', aspect='auto')
plt.subplot(122)
i = plt.imshow(my_image2, vmin=0, vmax=10, cmap='jet', aspect='auto')
plt.colorbar(i)

plt.show()
'''

from mpl_toolkits.axes_grid1 import ImageGrid

x = np.random.random(size=(10,10))


fig = plt.figure()
grid = ImageGrid(fig, 111,
                nrows_ncols = (1,2),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
                )

grid[0].imshow(x)
grid[0].axis('off')
grid[0].set_title('dog')

imc = grid[1].imshow(x, cmap='hot', interpolation='nearest')
grid[1].axis('off')
grid[1].set_title('dog')

plt.colorbar(imc, cax=grid.cbar_axes[0])

plt.show()
