
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

#include <iostream>
#include <math.h>

using namespace UPGMpp;
using namespace std;


int main (int argc, char* argv[])
{




/*------------------------------------------------------------------------------
 *
 *                               TRAINING!
 *
 *----------------------------------------------------------------------------*/

//    //
//    // Generate the types of nodes and edges
//    //

//    CTrainingDataSet trainingDataset;

//    size_t N_classes        = 2;
//    size_t N_nodeFeatures   = 3;
//    size_t N_edgeFeatures   = 3;

//    CNodeTypePtr simpleNodeType1Ptr( new CNodeType(N_classes, N_nodeFeatures) );

//    string label("Objects");
//    simpleNodeType1Ptr->setLabel( label );

//    CEdgeTypePtr simpleEdgeType1Ptr ( new CEdgeType(N_edgeFeatures, simpleNodeType1Ptr, simpleNodeType1Ptr) );

//    label = "Edge between two objects";
//    simpleNodeType1Ptr->setLabel( label );

//    trainingDataset.addNodeType( simpleNodeType1Ptr );
//    trainingDataset.addEdgeType( simpleEdgeType1Ptr );

//    //
//    // Create a set of training graphs
//    //

//    CGraph graph;
//    std::map<size_t,size_t> groundTruth;

//    Eigen::VectorXd nodeFeatures1(3);
//    nodeFeatures1 << 210,
//                     150,
//                     50;

//    Eigen::VectorXd nodeFeatures2(3);
//    nodeFeatures2 << 10,
//                     28,
//                     50;

//    Eigen::VectorXd edgeFeatures1(3);
//    edgeFeatures1 << 200,
//                     122,
//                     1;

//    CNodePtr nodePtr1 ( new CNode() );
//    nodePtr1->setType( simpleNodeType1Ptr );
//    nodePtr1->setFeatures( nodeFeatures1 );
//    graph.addNode( nodePtr1 );
//    groundTruth[ nodePtr1->getID() ] = 0;

//    CNodePtr nodePtr2 ( new CNode() );
//    nodePtr2->setType( simpleNodeType1Ptr );
//    nodePtr2->setFeatures( nodeFeatures2 );
//    graph.addNode( nodePtr2 );
//    groundTruth[ nodePtr2->getID() ] = 1;

//    CEdgePtr edgePtr1 ( new CEdge( nodePtr1,
//                                   nodePtr2,
//                                   simpleEdgeType1Ptr,
//                                   edgeFeatures1 ) );
//    graph.addEdge( edgePtr1 );

//    cout << "GRAPH: " << endl << graph << endl;

//    trainingDataset.addGraph( graph );
//    trainingDataset.addGraphGroundTruth( groundTruth );

////------------------------------------------------------------------------------

//    CGraph graph2;
//    std::map<size_t,size_t> groundTruth2;

//    Eigen::VectorXd nodeFeatures12(3);
//    nodeFeatures12 << 190,
//                     148,
//                     50;

//    Eigen::VectorXd nodeFeatures22(3);
//    nodeFeatures22 << 2,
//                     31,
//                     50;

//    Eigen::VectorXd edgeFeatures12(3);
//    edgeFeatures12 << 188,
//                     117,
//                     1;

//    CNodePtr nodePtr12 ( new CNode() );
//    nodePtr12->setType( simpleNodeType1Ptr );
//    nodePtr12->setFeatures( nodeFeatures12 );
//    graph2.addNode( nodePtr12 );
//    groundTruth2[ nodePtr12->getID() ] = 0;

//    CNodePtr nodePtr22 ( new CNode() );
//    nodePtr22->setType( simpleNodeType1Ptr );
//    nodePtr22->setFeatures( nodeFeatures22 );
//    graph2.addNode( nodePtr22 );
//    groundTruth2[ nodePtr22->getID() ] = 1;

//    CEdgePtr edgePtr12 ( new CEdge( nodePtr12,
//                                    nodePtr22,
//                                    simpleEdgeType1Ptr,
//                                    edgeFeatures12 ) );
//    graph2.addEdge( edgePtr12 );

//    cout << "GRAPH2: " << endl << graph2 << endl;

//    trainingDataset.addGraph( graph2 );
//    trainingDataset.addGraphGroundTruth( groundTruth2 );

////------------------------------------------------------------------------------

//    CGraph graph3;
//    std::map<size_t,size_t> groundTruth3;

//    Eigen::VectorXd nodeFeatures13(3);
//    nodeFeatures13 << 15,
//            25,
//            50;

//    Eigen::VectorXd nodeFeatures23(3);
//    nodeFeatures23 << 205,
//            155,
//            50;

//    Eigen::VectorXd edgeFeatures13(3);
//    edgeFeatures13 << 190,
//            120,
//            1;

//    CNodePtr nodePtr13 ( new CNode() );
//    nodePtr13->setType( simpleNodeType1Ptr );
//    nodePtr13->setFeatures( nodeFeatures13 );
//    graph3.addNode( nodePtr13 );
//    groundTruth3[ nodePtr13->getID() ] = 0;

//    CNodePtr nodePtr23 ( new CNode() );
//    nodePtr23->setType( simpleNodeType1Ptr );
//    nodePtr23->setFeatures( nodeFeatures23 );
//    graph3.addNode( nodePtr23 );
//    groundTruth3[ nodePtr23->getID() ] = 1;

//    CEdgePtr edgePtr13 ( new CEdge( nodePtr13,
//                                    nodePtr23,
//                                    simpleEdgeType1Ptr,
//                                    edgeFeatures13 ) );
//    graph3.addEdge( edgePtr13 );

//    cout << "GRAPH3: " << endl << graph3 << endl;

//    trainingDataset.addGraph( graph3 );
//    trainingDataset.addGraphGroundTruth( groundTruth3 );

////------------------------------------------------------------------------------


//    trainingDataset.train();

//// Test


//    graph.computePotentials();
//    graph2.computePotentials();
//    graph3.computePotentials();

//    /*cout << "GRAPH3: " << endl << graph << endl;

//    std::vector<size_t> results;
//    graph.decodeGreedy( results );

//    for ( size_t i = 0; i < results.size(); i++ )
//    {
//       std::cout << results[i] << std::endl;
//    }*/

//    cout << "New ICM Nigga!"<< endl;

//    CDecodeICM decodeICM;
//    CDecodeICMGreedy decodeICMGreedy;
//    CDecodeExact    decodeExact;

//    TOptions options;
//    options.maxIterations = 100;

//    std::map<size_t,size_t> resultsMap;

//    decodeICM.setOptions( options );
//    decodeICM.decode( graph2, resultsMap );

//    std::map<size_t,size_t>::iterator it;

//    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
//    {
//        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
//    }

//    cout << "New Greedy Nigga!"<< endl;

//    decodeICMGreedy.setOptions( options );
//    decodeICMGreedy.decode( graph3, resultsMap );

//    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
//    {
//        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
//    }

//    cout << "New Exact Nigga!"<< endl;

//    decodeExact.setOptions( options );
//    decodeExact.decode( graph, resultsMap );

//    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
//    {
//        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
//    }

//    cout << "New Exact with Mask Nigga!"<< endl;
//    std::map<size_t,std::vector<size_t> > mask;

//    std::vector<size_t> validValues;
//    validValues.push_back(1);

//    mask[0] = validValues;

//    decodeExact.setMask( mask );
//    decodeExact.decode( graph3, resultsMap );

//    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
//    {
//        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
//    }

    return 1;
}
