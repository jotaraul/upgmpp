
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
 *                         PREPARE TRAINING DATA
 *
 *----------------------------------------------------------------------------*/

    //
    // Generate the types of nodes and edges
    //

    CTrainingDataSet trainingDataset;

    size_t N_classes_type_1        = 2;
    size_t N_nodeFeatures_type_1   = 3;
    size_t N_edgeFeatures_type_1   = 3;

    size_t N_classes_type_2        = 3;
    size_t N_nodeFeatures_type_2   = 2;
    size_t N_edgeFeatures_type_2   = 2;

    // OBJECTS

    CNodeTypePtr simpleNodeType1Ptr( new CNodeType(N_classes_type_1,
                                                   N_nodeFeatures_type_1,
                                                   string("Objects") ) );

    CEdgeTypePtr simpleEdgeType1Ptr ( new CEdgeType(N_edgeFeatures_type_1,
                                                    simpleNodeType1Ptr,
                                                    simpleNodeType1Ptr,
                                                    string("Edges between two objects")) );

    // ROOMS

    CNodeTypePtr simpleNodeType2Ptr( new CNodeType(N_classes_type_2,
                                                   N_nodeFeatures_type_2,
                                                   string("Rooms") ) );

    CEdgeTypePtr simpleEdgeType2Ptr ( new CEdgeType(N_edgeFeatures_type_2,
                                                    simpleNodeType1Ptr,
                                                    simpleNodeType2Ptr,
                                                    "Edges between an object and a room") );


    Eigen::VectorXi typeOfEdgeFeatures( N_edgeFeatures_type_2 );
    typeOfEdgeFeatures << 1,1;

    trainingDataset.addNodeType( simpleNodeType1Ptr );
    trainingDataset.addEdgeType( simpleEdgeType1Ptr );
    trainingDataset.addNodeType( simpleNodeType2Ptr );
    trainingDataset.addEdgeType( simpleEdgeType2Ptr, typeOfEdgeFeatures );

    //
    // Create a set of training graphs
    //

/*--------------------------------- Graph 1 ----------------------------------*/


    CGraph graph;
    std::map<size_t,size_t> groundTruth;

    Eigen::VectorXd nodeFeatures1(3);
    nodeFeatures1 << 15, 30, 1;

    Eigen::VectorXd nodeFeatures2(2);
    nodeFeatures2 << 10, 1;

    Eigen::VectorXd edgeFeatures1(2);
    edgeFeatures1 << 1.5, 1;

    CNodePtr nodePtr2 ( new CNode( simpleNodeType2Ptr,
                                   nodeFeatures2,
                                   string("room-1") ) );

    CNodePtr nodePtr1 ( new CNode( simpleNodeType1Ptr,
                                   nodeFeatures1,
                                   string("object-1") ) );

    CEdgePtr edgePtr1 ( new CEdge( nodePtr2, nodePtr1,
                                   simpleEdgeType2Ptr,
                                   edgeFeatures1 ) ) ;

    graph.addNode( nodePtr1 );
    graph.addNode( nodePtr2 );

    graph.addEdge( edgePtr1 );

    groundTruth[ nodePtr1->getID() ] = 0;
    groundTruth[ nodePtr2->getID() ] = 0;

    trainingDataset.addGraph( graph );
    trainingDataset.addGraphGroundTruth( groundTruth );


/*--------------------------------- Graph 2 ----------------------------------*/


    CGraph graph2;
    std::map<size_t,size_t> groundTruth2;

    Eigen::VectorXd nodeFeatures12(3);
    nodeFeatures12 << 100, 70, 1;

    Eigen::VectorXd nodeFeatures22(2);
    nodeFeatures22 << 30, 1;

    Eigen::VectorXd edgeFeatures12(2);
    edgeFeatures12 << 3.33, 1;

    CNodePtr nodePtr12 ( new CNode( simpleNodeType1Ptr,
                                    nodeFeatures12 ) );

    CNodePtr nodePtr22 ( new CNode( simpleNodeType2Ptr,
                                    nodeFeatures22 ) );

    CEdgePtr edgePtr12 ( new CEdge( nodePtr12,
                                    nodePtr22,
                                    simpleEdgeType2Ptr,
                                    edgeFeatures12 ) );

    graph2.addNode( nodePtr12 );
    graph2.addNode( nodePtr22 );

    graph2.addEdge( edgePtr12 );

    groundTruth2[ nodePtr12->getID() ] = 1;
    groundTruth2[ nodePtr22->getID() ] = 1;

    //cout << "GRAPH2: " << endl << graph2 << endl;

    trainingDataset.addGraph( graph2 );
    trainingDataset.addGraphGroundTruth( groundTruth2 );


/*--------------------------------- Graph 3 ----------------------------------*/


    CGraph graph3;
    std::map<size_t,size_t> groundTruth3;

    Eigen::VectorXd nodeFeatures13(3);
    nodeFeatures13 << 16, 28, 1;

    Eigen::VectorXd nodeFeatures23(3);
    nodeFeatures23 << 95, 72, 1;

    Eigen::VectorXd edgeFeatures13(3);
    edgeFeatures13 << 69, 44, 1;

    CNodePtr nodePtr13 ( new CNode( simpleNodeType1Ptr,
                                    nodeFeatures13 ) );

    CNodePtr nodePtr23 ( new CNode( simpleNodeType1Ptr,
                                    nodeFeatures23 ) );

    CEdgePtr edgePtr13 ( new CEdge( nodePtr13,
                                    nodePtr23,
                                    simpleEdgeType1Ptr,
                                    edgeFeatures13 ) );

    graph3.addNode( nodePtr13 );
    graph3.addNode( nodePtr23 );

    graph3.addEdge( edgePtr13 );

    groundTruth3[ nodePtr13->getID() ] = 0;
    groundTruth3[ nodePtr23->getID() ] = 1;

    //cout << "GRAPH3: " << endl << graph3 << endl;

    trainingDataset.addGraph( graph3 );
    trainingDataset.addGraphGroundTruth( groundTruth3 );


/*------------------------------------------------------------------------------
 *
 *                               TRAINING!
 *
 *----------------------------------------------------------------------------*/

    UPGMpp::TTrainingOptions to;
    to.l2Regularization     = true;
    to.nodeLambda           = 10;
    to.edgeLambda           = 1;
    to.showTrainedWeights   = true;

    trainingDataset.setTrainingOptions( to );
    trainingDataset.train();

/*------------------------------------------------------------------------------
 *
 *                                TESTING!
 *
 *----------------------------------------------------------------------------*/

    graph.computePotentials();
    graph2.computePotentials();
    graph3.computePotentials();

    //
    // Build a graph to test the trained model
    //

    CGraph graph4;
    std::map<size_t,size_t> groundTruth4;

    Eigen::VectorXd nodeFeatures14(3);
    nodeFeatures14 << 17, 35, 1;

    Eigen::VectorXd nodeFeatures24(2);
    nodeFeatures24 << 12, 1;

    Eigen::VectorXd edgeFeatures14(2);
    edgeFeatures14 << 1.41, 1;

    CNodePtr nodePtr14 ( new CNode( simpleNodeType1Ptr,
                                    nodeFeatures14 ) );

    CNodePtr nodePtr24 ( new CNode( simpleNodeType2Ptr,
                                    nodeFeatures24 ) );


    CEdgePtr edgePtr14 ( new CEdge( nodePtr14,
                                    nodePtr24,
                                    simpleEdgeType2Ptr,
                                    edgeFeatures14 ) );

    graph4.addNode( nodePtr14 );
    graph4.addNode( nodePtr24 );

    graph4.addEdge( edgePtr14 );

    groundTruth4[ nodePtr14->getID() ] = 0;
    groundTruth4[ nodePtr24->getID() ] = 0;

    graph4.computePotentials();

    //cout << "GRAPH4: " << endl << graph4 << endl;

    //
    // Now compute the MAP and show the results
    //

    CDecodeICM          decodeICM;
    CDecodeICMGreedy    decodeICMGreedy;
    CDecodeExact        decodeExact;
    CDecodeLBP          decodeLBP;

    TInferenceOptions options;
    options.maxIterations   = 100;
    options.convergency     = 0.0001;


    cout << "ICM decoding"<< endl;

    std::map<size_t,size_t> resultsMap;

    decodeICM.setOptions( options );
    decodeICM.decode( graph4, resultsMap );

    std::map<size_t,size_t>::iterator it;

    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
    {
        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
    }


    cout << "ICM Greedy decoding"<< endl;

    decodeICMGreedy.setOptions( options );
    decodeICMGreedy.decode( graph4, resultsMap );

    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
    {
        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
    }


    cout << "Exact decoding"<< endl;

    decodeExact.setOptions( options );
    decodeExact.decode( graph4, resultsMap );

    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
    {
        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
    }


    cout << "LBP decoding" << endl;

    decodeLBP.setOptions( options );
    decodeLBP.decode( graph4, resultsMap );

    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
    {
        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
    }



    // We are ready to take a beer :)

    return 1;
}

