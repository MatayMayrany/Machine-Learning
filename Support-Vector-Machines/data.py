import numpy, random
import matplotlib.pyplot as plt

numpy.random.seed(100)
classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    numpy.random.rand(10, 2) * 0.2 + [-1.5, 0.5])
)
classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]), 
    -numpy.ones(classB.shape[0]))
)

N = inputs.shape[0] # Number of rows (samples)

permute=list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

def plot():
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal') # Force same scale on both axes
    plt.savefig('svmplot.pdf') # Save a copy in a file
    plt.show() # Show the plot on the screen

def plotContour(indicator):
    xgrid = numpy.linspace(-5, 5)
    ygrid = numpy.linspace(-4, 4)
    grid = numpy.array([[indicator(x, y) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid,
        (-1.0, 0.0, 1.0),
        colors = ('red', 'black', 'blue'),
        linewidths = (1, 3, 1))
