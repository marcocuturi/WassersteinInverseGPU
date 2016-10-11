# WassersteinInverseGPU
Computation of the derivative of the Loss ( WBarycenter(Dictionary,lambda), q)

This script was used to produce figure 11 in the following SIGGRAPH paper.

http://liris.cnrs.fr/~nbonneel/WassersteinBarycentricCoordinates/

In that experiment, we project a brain (3d tensor of ~200^3 voxels) onto a set of 14 brains in the Wasserstein sense.

That computation can be carried out by parallelizing computations of several GPUs. This is what the function in this package does.
