UPGM++
======

Undirected Probabilistic Graphical Models in C++ 
 
Copyright (C) 2014-2019 Jose Raul Ruiz Sarmiento 

University of Malaga (jotaraul@uma.es)
 
Personal webpage: http://mapir.isa.uma.es/jotaraul

Project webpage: http://mapir.isa.uma.es/work/upgmpp-library

ROS wrapper repository: https://github.com/MAPIRlab/upgmpp_wrapper

If you use UPGM++, please cite us by:

> @INPROCEEDINGS{Ruiz-Sarmiento-REACTS-2015,  <br/>
> author = {Ruiz-Sarmiento, J. R. and Galindo, Cipriano and Gonz{\'{a}}lez-Jim{\'{e}}nez, Javier},  <br/>
> title = {UPGMpp: a Software Library for Contextual Object Recognition},  <br/>
> booktitle = {3rd. Workshop on Recognition and Action for Scene Understanding (REACTS)},  <br/>
> year = {2015}  <br/>
> }

General information
-------------------

This is a standalone C++ library for Undirected Probabilistic Graphical Models. Some examples are provided for a better understanding of the library use. <strong>NEW!</strong> A ROS wrapper is now available [here](https://github.com/MAPIRlab/upgmpp_wrapper).

I want to thank Prof. Dr. Joachim Hertzberg and Mr. Martin Günther for the motivation and support that they gave me in the first steps of this library, which was started during my stay at the University of Osnabrück, Germany.

Prerequisites
-------------

Install dependencies:
- Eigen.
- Boost.
- liblbfgs:

        sudo apt-get install liblbfgs-dev

  That package is available starting with Ubuntu 12.10 Raring; if you're using
  Ubuntu Precise, go to http://packages.ubuntu.com and download the Raring
  package (or later). You also need the dependency `liblbfgs0`. From the command line:

        cd /tmp
        wget http://de.archive.ubuntu.com/ubuntu/pool/universe/libl/liblbfgs/liblbfgs0_1.10-5_amd64.deb     # replace amd64 by i386 if compiling on a 32 bit Ubuntu
        wget http://de.archive.ubuntu.com/ubuntu/pool/universe/libl/liblbfgs/liblbfgs-dev_1.10-5_amd64.deb  # replace amd64 by i386 if compiling on a 32 bit Ubuntu
        sudo dpkg -i liblbfgs0_1.10-5_amd64.deb liblbfgs-dev_1.10-5_amd64.deb


Compiling
---------

    mkdir build && cd build && cmake .. && make

Installation
------------

    sudo make install
