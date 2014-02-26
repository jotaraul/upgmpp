
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


#include "CGraph.hpp"

using namespace UPGMpp;
using namespace std;



/*------------------------------------------------------------------------------

                              computePotentials

------------------------------------------------------------------------------*/

void CGraph::computePotentials()
{
    // Method steps:
    //  1. Compute node potentials
    //  2. Compute edge potentials

    //
    //  1. Node potentials
    //

    std::vector<CNodePtr>::iterator it;

    //cout << "NODE POTENTIALS" << endl;

    for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
    {
        CNodePtr nodePtr = *it;
        // Get the node type
        size_t type = nodePtr->getType()->getID();

        // Compute the node potentials according to the node type and its
        // extracted features

        Eigen::VectorXd potentials = nodePtr->getType()->getWeights() * nodePtr->getFeatures();

        //cout << "+++++++++++++++++++++++++++++++" << endl;

        //cout << "Weights : " << nodePtr->getType()->getWeights() << endl;
        //cout << "Features: " << nodePtr->getFeatures().transpose() << endl;

        //cout << "Pre exp: " << potentials.transpose() << endl;
        potentials = potentials.array().exp();
        //cout << "Post exp: " << potentials.transpose() << endl;

        //std::cout << potentials << std::endl << "----------------" << std::endl;

        nodePtr->setPotentials( potentials );

    }

    //
    //  2. Edge potentials
    //

    std::vector<CEdgePtr>::iterator it2;

    //cout << "EDGE POTENTIALS" << endl;

    for ( it2 = m_edges.begin(); it2 != m_edges.end(); it2++ )
    {
        // Get the edge type, its extracted features and the number of them
        CEdgePtr edgePtr = *it2;
        size_t type = edgePtr->getType()->getID();

        Eigen::VectorXd v_feat = edgePtr->getFeatures();
        size_t num_feat = edgePtr->getType()->getWeights().size();

        // Compute the potential for each feature, and sum up them to obtain
        // the desired edge potential
        std::vector<Eigen::MatrixXd>    potentials_per_feat(num_feat);
        Eigen::MatrixXd potentials;


        for ( size_t feat = 0; feat < num_feat; feat++ )
        {
            potentials_per_feat.at(feat) = edgePtr->getType()->getWeights()[feat]*v_feat(feat);

            if ( !feat )
                potentials = potentials_per_feat[feat];
            else
                potentials += potentials_per_feat[feat];
        }

        potentials = potentials.array().exp();


        //std::cout <<  potentials << std::endl;
        //cout << "=======================================" << endl;

        edgePtr->setPotentials ( potentials );
    }

}

/*------------------------------------------------------------------------------

                          getUnnormalizedLikelihood

------------------------------------------------------------------------------*/

double CGraph::getUnnormalizedLogLikelihood( std::map<size_t,size_t> &classes)
{
    // TODO: This method is not robust against the deletion of nodes or edges

    double unlikelihood = 1;

    size_t N_nodes = m_nodes.size();
    size_t N_edges = m_edges.size();

    std::map<size_t,size_t>::iterator it;

    for ( it = classes.begin(); it != classes.end(); it++ )
    {
        CNodePtr node = getNodeWithID( it->first );
        unlikelihood *= node->getPotentials()(classes[node->getId()]);
    }

    for ( size_t index = 0; index < N_edges; index++ )
    {
        CEdgePtr edge = m_edges[index];
        CNodePtr n1, n2;
        edge->getNodes(n1,n2);
        size_t ID1 = n1->getId();
        size_t ID2 = n2->getId();

        if ( ID1 > ID2 )            
            unlikelihood *= edge->getPotentials()(classes[ID2],classes[ID1]);
        else
            unlikelihood *= edge->getPotentials()(classes[ID1],classes[ID2]);

    }


    unlikelihood = std::log( unlikelihood );

    return unlikelihood;
}
