# GNK-for-polyCT
MATLAB implementation of Gauss-Newton-Krylov for polychromatic CT reconstruction. Also includes L-BFGS and GD with Barzilai-Borwein steps as alternatives.

This code is complementary to the paper "Gauss-Newton-Krylov for Reconstruction ofPolychromatic X-ray CT Images" of Nathanael Six, Jens Renders, Jan Sijbers and Jan De Beenhouwer.

# paper abstract 
"Most lab-based X-ray sources are polychromatic, making the imaging process follow a non-linear model. However, widespread reconstruction algorithms, such as filtered back projection and the simultaneous iterative reconstruction technique, assume the reconstruction to be a linear problem, leading to artifacts in the reconstructions from polychromatic data. We propose to use quasi-Newton methods to minimize a polychromatic objective function, without the need of segmenting the image into different material regions. The objective function can also easily be extended with regularisation terms in a mathematically sound framework. We will show that these methods can outperform other statistical or algebraic reconstruction techniques. Reconstruction quality and projection error for reconstructions of both Monte-Carlo simulated data and experimental data are investigated. From the considered quasi-Newton methods, we find Gauss-Newton-Krylov to perform best. Compared to a recently proposed polychromatic algebraic reconstruction technique, quasi-Newton solvers reach a lower reconstruction error and have increased convergence speed."

# In this repository
Class AstraQuasiNewton and derived class AstraQuasiNewtonPoly implementing different quasi-Newton algorithms for polychromatic reconstructions. PiePhantomGATE.h5 contains a polychromatic Monte Carlo simulated CT dataset as well as the spectrum and material information needed for reconstruction.

# Dependencies
- MATLAB 
- Astra toolbox (https://github.com/astra-toolbox/astra-toolbox)
- Spot toolbox (https://github.com/mpf/spot)
