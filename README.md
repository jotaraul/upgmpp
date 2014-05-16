pgm++
=====

This is a standalone C++ library for Probabilistic Graphical Models. No ROS dependencies here.

Prerequisites
-------------

* Install dependencies:

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
