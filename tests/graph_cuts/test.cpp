/*---------------------------------------------------------------------------*
 |                               UPGM++                                      |
 |                   Undirected Graphical Models in C++                      |
 |                                                                           |
 |              Copyright (C) 2014 Jose Raul Ruiz Sarmiento                  |
 |                 University of Malaga (jotaraul@uma.es)                    |
 |                         University of Osnabruk                            |
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


#include "base.hpp"
#include "training.hpp"
#include "decoding.hpp"
#include "inference.hpp"

#include <iostream>
#include <math.h>

// C++ program for implementation of Ford Fulkerson algorithm
#include <queue>


using namespace UPGMpp;
using namespace Eigen;
using namespace std;


// Driver program to test above functions
int main()
{

    MatrixXd graph(6,6);

    graph <<  0.0, 16.0, 13.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 10.0, 12.0, 0.0, 0.0,
                0.0, 4.0, 0.0, 0.0, 14.0, 0.0,
                0.0, 0.0, 9.0, 0.0, 0.0, 20.0,
                0.0, 0.0, 0.0, 7.0, 0.0, 4.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0;


    size_t N_nodes = graph.cols();
    VectorXi cut(N_nodes);
    cut.setZero();

    cout << "The maximum possible flow is " << fordFulkerson(graph, 0, 5, cut) << endl;

    cout << "The minimun cut is: ";

    for ( int i=0; i<N_nodes; i++ )
        cout << cut(i) << " ";

    cout << endl;

    return 0;
}
