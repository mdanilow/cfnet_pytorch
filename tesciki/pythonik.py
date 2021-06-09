import numpy as np
import torch
import os
import sys
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)
TEST_PATH = '/home/magister/CF_tracking/cfnet_pytorch/tesciki'

import training.models as mdl


def generate_gaussian(size, sigma):

    xg, yg = np.meshgrid(range(size), range(size))
    half = size // 2
    xg = np.mod(xg + half, size) - half
    yg = np.mod(yg + half, size) - half

    y = np.exp(-(np.square(xg) + np.square(yg)) / (2 * sigma**2))

    plotwindow(y)
    
    return y
    

def plotwindow(data, title='plot'):

    nx, ny = data.shape
    x = range(nx)
    y = range(ny)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data)

    plt.title(title)
    plt.show(block=False)
    plt.show()


# generate_gaussian(49, 2)
# sys.exit()

cfblock = mdl.CFblock(57)

# test_tens = torch.rand(1, 32, 57, 57).numpy()
# np.save('tesciki/testowy_tensor', test_tens)
testtensor = np.load('/home/magister/CF_tracking/cfnet_pytorch/tesciki/testowy_tensor.npy')
testtensor = torch.tensor(testtensor).to('cuda')

# target test
targettf = np.genfromtxt(join(TEST_PATH, 'targettf.csv'), delimiter=',')
targettorch = np.genfromtxt(join(TEST_PATH, 'targettorch.csv'), delimiter=',')
targetdiffs = np.abs(targettf - targettorch)
plotwindow(targettf, 'tf')
plotwindow(targettorch, 'torch')
print('targeterror:', np.sum(targetdiffs))

# window test
windowtf = np.genfromtxt(join(TEST_PATH, 'windowtf.csv'), delimiter=',')
windowtorch = np.genfromtxt(join(TEST_PATH, 'windowtorch.csv'), delimiter=',')
windowdiffs = np.abs(windowtf - windowtorch)
# plotwindow(windowtf, 'tf')
# plotwindow(windowtorch, 'torch')
print('windowerror:', np.sum(windowdiffs))

# final test
resulttf = np.load(join(TEST_PATH, 'outtf.npy'))
resulttf = np.transpose(resulttf, (2, 0, 1))
result = cfblock(testtensor)[0].cpu().numpy()
resultdiffs = np.abs(result - resulttf)

print('cum resulterror:', np.sum(resultdiffs))
print('max resulterror:', np.max(resultdiffs))