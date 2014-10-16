 *---------------------------------------------------------------------------*
 |                               UPGM++                                      |
 |                   Undirected Graphical Models in C++                      |
 |                                                                           |
 |              Copyright (C) 2014 Jose Raul Ruiz Sarmiento                  |
 |                 University of Malaga (jotaraul@uma.es)                    |
 |                                                                           |
 |   This program is free software: you can redistribute it and/or modify    |
 |   it under the terms of the GNU General Public License as published by    |
 |   the Free Software Foundation, either version 3 of the License, or       |
 |   (at your option) any later version.                                     |
 |                                                                           |
 |   This program is distributed in the hope that it will be useful,         |
 |   but WITHOUT ANY WARRANTY; without even the implied warranty of          |
 |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
 |   GNU General Public License for more details.                            |
 |   <http://www.gnu.org/licenses/>                                          |
 |                                                                           |
 *---------------------------------------------------------------------------*/

General information
-------------------

This is a standalone C++ library for Undirected Probabilistic Graphical Models. Some examples are provided for a better understanding of the library use.

I want to thank Prof. Dr. Joachim Hertzberg and Mr. Martin Günter for the motivation and support that they gave me in the first steps of this library, which was started during my stay at the University of Osnabrück, Germany.

Prerequisites
-------------

Install dependencies:
	* Eigen.
	* Boost.
	* liblbfgs.

- liblbfgs

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
