pgm++
=====

This is a standalone C++ library for Probabilistic Graphical Models. No ROS dependencies here.

Prerequisites
-------------

* Install dependencies:

    sudo apt-get install liblbfgs-dev

  That package is available starting with Ubuntu 12.10 Raring; if you're using
  Ubuntu Precise, go to http://packages.ubuntu.com and download the Raring
  package (or later). You also need the dependency `liblbfgs0`.

Compiling
---------

    mkdir build && cd build && cmake .. && make
