
/*---------------------------------------------------------------------------*
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


#include "CGraph.hpp"

using namespace UPGMpp;
using namespace std;



///*------------------------------------------------------------------------------

//                              computePotentials

//------------------------------------------------------------------------------*/

//void CGraph::computePotentials()
//{
//    // Method steps:
//    //  1. Compute node potentials
//    //  2. Compute edge potentials

//    //
//    //  1. Node potentials
//    //

//    std::vector<CNodePtr>::iterator it;

//    //cout << "NODE POTENTIALS" << endl;

//    for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
//    {
//        CNodePtr nodePtr = *it;

//        if ( !nodePtr->finalPotentials() )
//        {
//            // Get the node type
//            size_t type = nodePtr->getType()->getID();

//            // Compute the node potentials according to the node type and its
//            // extracted features

//            Eigen::VectorXd potentials = nodePtr->getType()->computePotentials( nodePtr->getFeatures() );

//            // Apply the node class multipliers
//            potentials = potentials.cwiseProduct( nodePtr->getClassMultipliers() );

//            /*Eigen::VectorXd fixed = nodePtr->getFixed();

//            potentials = potentials.cwiseProduct( fixed );*/

//            nodePtr->setPotentials( potentials );
//        }

//    }

//    //
//    //  2. Edge potentials
//    //

//    std::vector<CEdgePtr>::iterator it2;

//    //cout << "EDGE POTENTIALS" << endl;

//    for ( it2 = m_edges.begin(); it2 != m_edges.end(); it2++ )
//    {
//        CEdgePtr edgePtr = *it2;

//        Eigen::MatrixXd potentials
//                = edgePtr->getType()->computePotentials( edgePtr->getFeatures() );

//        edgePtr->setPotentials ( potentials );
//    }

//}

///*------------------------------------------------------------------------------

//                          getUnnormalizedLikelihood

//------------------------------------------------------------------------------*/

//double CGraph::getUnnormalizedLogLikelihood( std::map<size_t,size_t> &classes)
//{

//    double unlikelihood = 1;

//    size_t N_nodes = m_nodes.size();
//    size_t N_edges = m_edges.size();

//    std::map<size_t,size_t>::iterator it;

//    for ( it = classes.begin(); it != classes.end(); it++ )
//    {
//        CNodePtr node = getNodeWithID( it->first );
//        unlikelihood *= node->getPotentials()(classes[node->getID()]);
//    }

//    for ( size_t index = 0; index < N_edges; index++ )
//    {
//        CEdgePtr edge = m_edges[index];
//        CNodePtr n1, n2;
//        edge->getNodes(n1,n2);
//        size_t ID1 = n1->getID();
//        size_t ID2 = n2->getID();

//        if ( ID1 > ID2 )
//            unlikelihood *= edge->getPotentials()(classes[ID2],classes[ID1]);
//        else
//            unlikelihood *= edge->getPotentials()(classes[ID1],classes[ID2]);

//    }

//    unlikelihood = std::log( unlikelihood );

//    return unlikelihood;
//}

