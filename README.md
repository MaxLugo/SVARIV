# SVARIV in python 

(This repository is a python implementation of the original repository in matlab)

This repository contains a python suite to construct weak-instrument robust confidence intervals for impulse response coefficients in Structural Vector Autoregressions identified with an external instrument. See "Inference in Structural Vector Autoregressions identified by an external instrument" by J.L Montiel Olea, J. H. Stock, and M. W. Watso…

The original repository is the following: https://github.com/jm4474/SVARIV

The original paper can be download from: http://www.joseluismontielolea.com/papers.html 

Folder:

data = required data from the paper (taken from the original data).  

Files:

SVARIV folder.

Contains the related functions for the implementation. The What, wald test, Gamma_hat, irf procedures and plot functions are inside the file.
	
	- IRF puntual estimation function

	- MA representarion and gradient estimation function

	- WHat, wald test and Gamma function

	- Confidence set estimations 


example.py
The file will call SVARIV functions and replicates the results. file format .py (native python)

example.ipynb
The file will call SVARIV functions and replicates the results. file format .ipynb (jupyter notebook)

Both files example.* have the same content. The python file replicates Figure 1 and 2 and related results of the Oil 
example exogenous instrument.

requirements.txt
Libraries required to run the python implementation.

runtime.txt
Python version to be use (in this case was tested in python-3.7.6)
But it should run in python 3+

figure1_matlabvspython_validation.xlsx
The file compare the matlab (original) results and the python results (replicaiton). This is only made for the first figure. It is also accomplished for the second, but it is not presented. The file was made for validation porpueses for the plug-in confidense sets (CS-plug-in) and also it is possible to validate the CS-AR.


## Jobs to be done

1. F test is not presented (test for weak instruments, 2013 paper of the same author). 
2. Monte Carlo excercises (it is not created)














