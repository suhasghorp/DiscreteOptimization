import numpy

n=10
a = numpy.random.rand(n, 3)
b = a
out0 = numpy.array([a, b])
out1 = numpy.array([a.T, b.T])

temp =  numpy.sqrt(numpy.sum((a - b) ** 2, axis=1))

print("")