
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


#include "CEdgeType.hpp"

using namespace UPGMpp;
using namespace std;


/*------------------------------------------------------------------------------

                              linearModelEdge

------------------------------------------------------------------------------*/

Eigen::MatrixXd UPGMpp::linearModelEdge(vector<Eigen::MatrixXd> &weights, Eigen::VectorXd &features)
{
    size_t N_feat = weights.size();
    // Compute the potential for each feature, and sum up them to obtain
    // the desired edge potential
    std::vector<Eigen::MatrixXd>    potentials_per_feat(N_feat);
    Eigen::MatrixXd                 potentials;


    for ( size_t feat = 0; feat < N_feat; feat++ )
    {
        potentials_per_feat.at(feat) = weights[feat]*features(feat);

        if ( !feat )
            potentials = potentials_per_feat[feat];
        else
            potentials += potentials_per_feat[feat];
    }

    potentials = potentials.array().exp();

    return potentials;
}
