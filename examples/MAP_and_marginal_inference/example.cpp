
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
#include "inference_MAP.hpp"
#include "inference_marginal.hpp"

#include <iostream>
#include <math.h>

// include headers that implement a archive in simple text format
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace UPGMpp;
using namespace std;


/*---------------------------------------------------------------------------*
 *
 * This example aims to show how to use compute the MAP (Maximum a Posteriori)
 * and the marginal probabilities of nodes and edges. In this library, we will
 * refer to the MAP computation as decoding, and to the marginal propabilities
 * computation as inference. Although both are different types of probabilistic
 * inference, this will help to easily differentiate these tasks.
 *
 * The work flow in this example is as follow:
 * 1. Preparation of training data.
 * 2. Training of PGM parameters.
 * 3. Perform the decode taks with some of the implemented methods for that.
 * 4. Perform inference.
 *
 *---------------------------------------------------------------------------*/

void showResults(  std::map<size_t,size_t> &resultsMap );

int main (int argc, char* argv[])
{

    cout << endl;
    cout << "      MAP AND MARGINAL INFERENCE EXAMPLE";
    cout << endl << endl;

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

    // OBJECTS

    CNodeTypePtr simpleNodeType1Ptr( new CNodeType(N_classes_type_1,
                                                   N_nodeFeatures_type_1,
                                                   string("Objects") ) );

    CEdgeTypePtr simpleEdgeType1Ptr ( new CEdgeType(N_edgeFeatures_type_1,
                                                    simpleNodeType1Ptr,
                                                    simpleNodeType1Ptr,
                                                    string("Edges between two objects")) );


    trainingDataset.addNodeType( simpleNodeType1Ptr );

    Eigen::VectorXi typeOfEdgeFeatures(N_edgeFeatures_type_1);
    typeOfEdgeFeatures << 1,1,1;
    trainingDataset.addEdgeType( simpleEdgeType1Ptr, typeOfEdgeFeatures );

    //
    // Create a set of training graphs
    //

/*--------------------------------- Graph 1 ----------------------------------*/


    CGraph graph1;
    std::map<size_t,size_t> groundTruth1;

    Eigen::VectorXd nodeFeatures11(3), nodeFeatures21(3), edgeFeatures11(3);

    nodeFeatures11 << 16, 22, 1;
    nodeFeatures21 << 12, 29, 1;
    edgeFeatures11 << 4, 7, 1;

    CNodePtr nodePtr11 ( new CNode( simpleNodeType1Ptr, nodeFeatures11 ) );

    CNodePtr nodePtr21 ( new CNode( simpleNodeType1Ptr, nodeFeatures21 ) );

    CEdgePtr edgePtr11 ( new CEdge( nodePtr11, nodePtr21,
                                    simpleEdgeType1Ptr, edgeFeatures11 ) );

    graph1.addNode( nodePtr11 );
    graph1.addNode( nodePtr21 );

    graph1.addEdge( edgePtr11 );

    groundTruth1[ nodePtr11->getID() ] = 0;
    groundTruth1[ nodePtr21->getID() ] = 0;

    trainingDataset.addGraph( graph1 );
    trainingDataset.addGraphGroundTruth( groundTruth1 );

/*--------------------------------- Graph 2 ----------------------------------*/


    CGraph graph2;
    std::map<size_t,size_t> groundTruth2;

    Eigen::VectorXd nodeFeatures12(3), nodeFeatures22(3), edgeFeatures12(3);

    nodeFeatures12 << 91, 78, 1;
    nodeFeatures22 << 20, 24, 1;
    edgeFeatures12 << 71, 54, 1;

    CNodePtr nodePtr12 ( new CNode( simpleNodeType1Ptr, nodeFeatures12 ) );

    CNodePtr nodePtr22 ( new CNode( simpleNodeType1Ptr, nodeFeatures22 ) );

    CEdgePtr edgePtr12 ( new CEdge( nodePtr12, nodePtr22,
                                    simpleEdgeType1Ptr, edgeFeatures12 ) );

    graph2.addNode( nodePtr12 );
    graph2.addNode( nodePtr22 );

    graph2.addEdge( edgePtr12 );

    groundTruth2[ nodePtr12->getID() ] = 1;
    groundTruth2[ nodePtr22->getID() ] = 0;

    trainingDataset.addGraph( graph2 );
    trainingDataset.addGraphGroundTruth( groundTruth2 );


/*--------------------------------- Graph 3 ----------------------------------*/


    CGraph graph3;
    std::map<size_t,size_t> groundTruth3;

    Eigen::VectorXd nodeFeatures13(3), nodeFeatures23(3), edgeFeatures13(3);

    nodeFeatures13 << 16, 28, 1;    
    nodeFeatures23 << 95, 72, 1;
    edgeFeatures13 << 69, 44, 1;

    CNodePtr nodePtr13 ( new CNode( simpleNodeType1Ptr, nodeFeatures13 ) );

    CNodePtr nodePtr23 ( new CNode( simpleNodeType1Ptr, nodeFeatures23 ) );

    CEdgePtr edgePtr13 ( new CEdge( nodePtr13, nodePtr23,
                                    simpleEdgeType1Ptr, edgeFeatures13 ) );

    graph3.addNode( nodePtr13 );
    graph3.addNode( nodePtr23 );

    graph3.addEdge( edgePtr13 );

    groundTruth3[ nodePtr13->getID() ] = 0;
    groundTruth3[ nodePtr23->getID() ] = 1;

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
    to.showTrainedWeights   = false;
    to.showTrainingProgress = false;

    cout << "------------------------------------------------------" << endl;
    cout << "                      TRAINING" << endl;
    cout << "------------------------------------------------------" << endl;

    trainingDataset.setTrainingOptions( to );
    trainingDataset.train();


/*------------------------------------------------------------------------------
 *
 *              PREPARE A GRAPH FOR PERFORMING DECODING AND INFERENCE
 *
 *----------------------------------------------------------------------------*/

    CGraph testGraph;
    std::map<size_t,size_t> groundTruth;

    Eigen::VectorXd nodeFeatures1(3), nodeFeatures2(3), edgeFeatures1(3);

    nodeFeatures1 << 12, 21, 1;
    nodeFeatures2 << 88, 77, 1;
    edgeFeatures1 << 76, 56, 1;

    CNodePtr nodePtr1 ( new CNode( simpleNodeType1Ptr, nodeFeatures1 ) );

    CNodePtr nodePtr2 ( new CNode( simpleNodeType1Ptr, nodeFeatures2 ) );

    CEdgePtr edgePtr1 ( new CEdge( nodePtr1, nodePtr2,
                                    simpleEdgeType1Ptr, edgeFeatures1 ) );

    testGraph.addNode( nodePtr1 );
    testGraph.addNode( nodePtr2 );

    testGraph.addEdge( edgePtr1 );

    groundTruth[ nodePtr1->getID() ] = 0;
    groundTruth[ nodePtr2->getID() ] = 1;

    testGraph.computePotentials();



/*------------------------------------------------------------------------------
 *
 *                               DECODING!
 *
 *----------------------------------------------------------------------------*/

    cout << endl;
    cout << "------------------------------------------------------" << endl;
    cout << "                      DECODING" << endl;
    cout << "------------------------------------------------------" << endl;


    CICMInferenceMAP       decodeICM;
    CICMGreedyInferenceMAP decodeICMGreedy;
    CExactInferenceMAP     decodeExact;
    CLBPInferenceMAP       decodeLBP;
    CAlphaExpansionInferenceMAP decodeAlphaExpansion;

    TInferenceOptions options;
    options.maxIterations = 100;

    std::map<size_t,size_t> resultsMap;

    std::cout << "           RESULTS ICM " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    decodeICM.setOptions( options );
    decodeICM.infer( testGraph, resultsMap );

    showResults( resultsMap );

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "         RESULTS ICM GREEDY" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    decodeICMGreedy.setOptions( options );
    decodeICMGreedy.infer( testGraph, resultsMap );

    showResults( resultsMap );

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "              EXACT " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    decodeExact.setOptions( options );
    decodeExact.infer( testGraph, resultsMap );

    showResults( resultsMap );

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "          ALPHA_EXPANSION " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    decodeAlphaExpansion.setOptions( options );
    decodeAlphaExpansion.infer( testGraph, resultsMap );

    showResults( resultsMap );

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "              LBP " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    options.convergency = 0.0001;
    options.maxIterations = 10;

    decodeLBP.setOptions( options );
    decodeLBP.infer( testGraph, resultsMap );

    showResults( resultsMap );

/*------------------------------------------------------------------------------
 *
 *                               INFERENCE!
 *
 *----------------------------------------------------------------------------*/

    cout << endl;
    cout << "------------------------------------------------------" << endl;
    cout << "                      INFERENCE" << endl;
    cout << "------------------------------------------------------" << endl;


    std::cout << "     LOOPY BELIEF PROPAGATION " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    CLBPInferenceMarginal LBPInference;

    map<size_t,VectorXd> nodeBeliefs;
    map<size_t,MatrixXd> edgeBeliefs;
    double logZ;

    LBPInference.infer( testGraph, nodeBeliefs, edgeBeliefs, logZ );

    map<size_t,VectorXd>::iterator itNodeBeliefs;
    for ( itNodeBeliefs = nodeBeliefs.begin(); itNodeBeliefs != nodeBeliefs.end(); itNodeBeliefs++ )
    {
        std::cout << "Node id " << itNodeBeliefs->first << " beliefs (marginals): "
                  << endl << itNodeBeliefs->second << std::endl << std::endl;
    }

    map<size_t,MatrixXd>::iterator itEdgeBeliefs;
    for ( itEdgeBeliefs = edgeBeliefs.begin(); itEdgeBeliefs != edgeBeliefs.end(); itEdgeBeliefs++ )
    {
        std::cout << "Edge id " << itEdgeBeliefs->first << " beliefs (marginals): "
                  << endl << itEdgeBeliefs->second << std::endl << std::endl;
    }

    cout << "Log Z : " << logZ << endl;


    std::cout << "-----------------------------------" << std::endl;
    std::cout << "    TREE REPARAMETRIZATION BP " << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    CTRPBPInferenceMarginal TRPBPInference;

    TRPBPInference.infer( testGraph, nodeBeliefs, edgeBeliefs, logZ );

    for ( itNodeBeliefs = nodeBeliefs.begin(); itNodeBeliefs != nodeBeliefs.end(); itNodeBeliefs++ )
    {
        std::cout << "Node id " << itNodeBeliefs->first << " beliefs (marginals): "
                  << endl << itNodeBeliefs->second << std::endl << std::endl;
    }

    for ( itEdgeBeliefs = edgeBeliefs.begin(); itEdgeBeliefs != edgeBeliefs.end(); itEdgeBeliefs++ )
    {
        std::cout << "Edge id " << itEdgeBeliefs->first << " beliefs (marginals): "
                  << endl << itEdgeBeliefs->second << std::endl << std::endl;
    }

    cout << "Log Z : " << logZ << endl;

    // We are ready to take a beer :)

    return 0;
}


/** Auxiliar function to show the results of a decoding task.
 */
void showResults(  std::map<size_t,size_t> &resultsMap )
{
    std::map<size_t,size_t>::iterator it;

    for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
    {
        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
    }
}
