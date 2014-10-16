
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


#include "base.hpp"
#include "training.hpp"
#include "decoding.hpp"

#include <iostream>
#include <math.h>

using namespace UPGMpp;
using namespace std;


/*---------------------------------------------------------------------------*
 *
 * In this example how to train a PGM is shown. For that, first three training
 * examples, i.e. graphs, are build. Then, the parameters of the PGM are learnt
 * through training. These parameters correspond with the weights of the node
 * types and edge types used, which are used to perform inference tasks in new
 * graphs. The progress of the training process as well as its result are
 * prompted.
 *
 *---------------------------------------------------------------------------*/

int main (int argc, char* argv[])
{


    cout << endl;
    cout << "              " << "TRAINING EXAMPLE";
    cout << endl << endl;


/*------------------------------------------------------------------------------
 *
 *                      PREPARATION OF TRAINING SAMPLES
 *
 *----------------------------------------------------------------------------*/

    //
    // Generate the types of nodes and edges
    //

    CTrainingDataSet trainingDataset;

    size_t N_classes        = 2;
    size_t N_nodeFeatures   = 3;
    size_t N_edgeFeatures   = 3;

    CNodeTypePtr simpleNodeType1Ptr( new CNodeType(N_classes, N_nodeFeatures) );

    CEdgeTypePtr simpleEdgeType1Ptr ( new CEdgeType(N_edgeFeatures,
                                                    simpleNodeType1Ptr,
                                                    simpleNodeType1Ptr) );

    trainingDataset.addNodeType( simpleNodeType1Ptr );
    trainingDataset.addEdgeType( simpleEdgeType1Ptr );

    //
    // Create a set of training graphs
    //

//------------------------------------------------------------------------------

    CGraph graph;
    std::map<size_t,size_t> groundTruth;

    Eigen::VectorXd nodeFeatures1(3);
    nodeFeatures1 << 210, 150, 50;

    Eigen::VectorXd nodeFeatures2(3);
    nodeFeatures2 << 10, 28, 50;

    Eigen::VectorXd edgeFeatures1(3);
    edgeFeatures1 << 200, 122, 1;

    CNodePtr nodePtr1 ( new CNode( simpleNodeType1Ptr, nodeFeatures1 ) );

    CNodePtr nodePtr2 ( new CNode( simpleNodeType1Ptr, nodeFeatures2 ) );

    CEdgePtr edgePtr1 ( new CEdge( nodePtr1, nodePtr2,
                                   simpleEdgeType1Ptr, edgeFeatures1 ) );

    graph.addNode( nodePtr1 );
    graph.addNode( nodePtr2 );
    graph.addEdge( edgePtr1 );

    groundTruth[ nodePtr1->getID() ] = 0;
    groundTruth[ nodePtr2->getID() ] = 1;

    trainingDataset.addGraph( graph );
    trainingDataset.addGraphGroundTruth( groundTruth );

//------------------------------------------------------------------------------

    CGraph graph2;
    std::map<size_t,size_t> groundTruth2;

    Eigen::VectorXd nodeFeatures12(3);
    nodeFeatures12 << 190, 148, 50;

    Eigen::VectorXd nodeFeatures22(3);
    nodeFeatures22 << 2, 31, 50;

    Eigen::VectorXd edgeFeatures12(3);
    edgeFeatures12 << 188, 117, 1;

    CNodePtr nodePtr12 ( new CNode( simpleNodeType1Ptr, nodeFeatures12 ) );

    CNodePtr nodePtr22 ( new CNode( simpleNodeType1Ptr, nodeFeatures22 ) );

    CEdgePtr edgePtr12 ( new CEdge( nodePtr12, nodePtr22,
                                    simpleEdgeType1Ptr, edgeFeatures12 ) );

    graph2.addNode( nodePtr12 );
    graph2.addNode( nodePtr22 );
    graph2.addEdge( edgePtr12 );

    groundTruth2[ nodePtr12->getID() ] = 0;
    groundTruth2[ nodePtr22->getID() ] = 1;

    trainingDataset.addGraph( graph2 );
    trainingDataset.addGraphGroundTruth( groundTruth2 );

//------------------------------------------------------------------------------

    CGraph graph3;
    std::map<size_t,size_t> groundTruth3;

    Eigen::VectorXd nodeFeatures13(3);
    nodeFeatures13 << 15, 25, 50;

    Eigen::VectorXd nodeFeatures23(3);
    nodeFeatures23 << 205, 155, 50;

    Eigen::VectorXd edgeFeatures13(3);
    edgeFeatures13 << 190, 120, 1;

    CNodePtr nodePtr13 ( new CNode( simpleNodeType1Ptr, nodeFeatures13 ) );

    CNodePtr nodePtr23 ( new CNode( simpleNodeType1Ptr, nodeFeatures23 ) );

    CEdgePtr edgePtr13 ( new CEdge( nodePtr13, nodePtr23,
                                    simpleEdgeType1Ptr, edgeFeatures13 ) );

    graph3.addNode( nodePtr23 );
    graph3.addNode( nodePtr13 );
    graph3.addEdge( edgePtr13 );

    groundTruth3[ nodePtr13->getID() ] = 1;
    groundTruth3[ nodePtr23->getID() ] = 0;

    trainingDataset.addGraph( graph3 );
    trainingDataset.addGraphGroundTruth( groundTruth3 );


/*------------------------------------------------------------------------------
 *
 *                      PREPARATION OF TRAINING SAMPLES
 *
 *----------------------------------------------------------------------------*/

    cout << "---------------------------------------------" << endl;
    cout << "                 TRAINING " << endl;
    cout << "---------------------------------------------" << endl << endl;

    TTrainingOptions to;
    to.l2Regularization     = true;
    to.nodeLambda           = 10;
    to.edgeLambda           = 1;
    to.showTrainedWeights   = true;
    to.showTrainingProgress = true;

    trainingDataset.setTrainingOptions( to );

    trainingDataset.train();

    // :)


    return 1;
}
