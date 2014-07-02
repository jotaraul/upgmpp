
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

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <math.h>
#include <map>

using namespace UPGMpp;
using namespace std;
using namespace Eigen;

// Type definitions for a easier gaussian random number generation
typedef boost::normal_distribution<double> NormalDistribution;
typedef boost::mt19937 RandomGenerator;
typedef boost::variate_generator<RandomGenerator&, \
                        NormalDistribution> GaussianGenerator;


/*---------------------------------------------------------------------------*
 *
 * This example illustrates how to create synthetic training data in a easy way,
 * how this data is used to train a PGM, and how to generate test samples to
 * check the PGM performance.
 *
 *---------------------------------------------------------------------------*/

int main (int argc, char* argv[])
{

    cout << endl;
    cout << "     " << "WORKING WITH SYNTHETIC DATA EXAMPLE";
    cout << endl << endl;

/*------------------------------------------------------------------------------
 *
 *                     PREPARE SYNTHETIC TRAINING DATA
 *
 *----------------------------------------------------------------------------*/

    // Initiate Random Number generator with current time
    static RandomGenerator rng(static_cast<unsigned> (time(0)));

    CTrainingDataSet trainingDataset;

    /* Suppose the following object classes:
     *  0: bin
     *  1: table
     *  2: book
     */

    size_t N_classes = 3;

    //
    // Node features
    //

    /* And that we are going to use three features for characterize them:
     *  0: centroid height
     *  1: size
     *  2: bias
     */
    size_t N_nodeFeatures = 3;

    // Specification of the mean value for each feature and class
    MatrixXd classesFeatMean(N_classes,N_nodeFeatures);
    classesFeatMean <<  0.15,   0.3,      1,
                        0.75,   1,        1,
                        0.85,   0.20,     1;

    // Specification of the stdv
    MatrixXd classesFeatStd(N_classes,N_nodeFeatures);
    classesFeatStd <<  0.05,   0.02,   0,
                        0.02,   0.3,   0,
                        0.02,   0.05,  0;

    //
    // Edge features
    //
    size_t N_edgeFeatures = 5;

    VectorXd frequencyOfOcurrence(N_classes);
    frequencyOfOcurrence << 100, 100, 100;

    MatrixXd relationsFrequency(N_classes,N_classes);
    relationsFrequency << 0,    100,    0,
                          100,  0,      100,
                          0,    100,    0;

    //
    // Relational features
    //
    MatrixXd on_top_of( N_classes, N_classes );
             // bin table book
    on_top_of << 0,   0,   0,  // bin
                 0,   0,   0,  // table
                 0,   1,   0;  // book


    //
    // Create the node types
    //
    CNodeTypePtr nodeType1Ptr( new CNodeType(N_classes, N_nodeFeatures) );

    string label("Objects");
    nodeType1Ptr->setLabel( label );
    trainingDataset.addNodeType( nodeType1Ptr );

    //
    // Create the edge types
    //
    CEdgeTypePtr edgeType1Ptr ( new CEdgeType(N_edgeFeatures, nodeType1Ptr, nodeType1Ptr) );

    label = "Edge between two objects";
    edgeType1Ptr->setLabel( label );

    // This vector set the type of the different edge features.
    // 0 means that the edge feature is symetric.
    // 1 means that it is asymetric.
    // 2 means that it is just the transpose of the previous feature.
    VectorXi typeOfEdgeFeatures( N_edgeFeatures );
    typeOfEdgeFeatures << 0, 0, 1, 1, 0;

    trainingDataset.addEdgeType( edgeType1Ptr, typeOfEdgeFeatures );

    // Number of scenarios
    size_t N_syntheticTrainingSamples = 200;

    //
    // Create the scenarios!
    //
    for ( size_t N_sample = 0; N_sample < N_syntheticTrainingSamples; N_sample++ )
    {
        // Create the graph for the scenario
        CGraph graph;
        std::map<size_t,size_t> groundTruth;

        //
        // Create the nodes of the graph (objects in the scenario)
        //

        vector<CNodePtr> nodes;
        map<size_t,size_t> classesConsidered;

        for ( size_t count = 0; count < N_classes; count++ )
        {
            // Decide the class of the next object
            size_t N_class;
            size_t classToCheck = rand() % N_classes;

            while ( classesConsidered[classToCheck] == 1 )
                classToCheck = rand() % N_classes;

            N_class = classToCheck;
            classesConsidered[N_class] = 1;

            //cout << "Object class" << N_class << endl;

            // Decide if the object of this class appear in the scenario
            size_t random = rand() % 100;

            if ( frequencyOfOcurrence(N_class) > random )
            {
                // Generate node features
                VectorXd nodeFeatures(N_nodeFeatures);

                for ( size_t nodeFeat = 0; nodeFeat < N_nodeFeatures; nodeFeat++ )
                {
                    NormalDistribution gaussian_dist(
                                classesFeatMean(N_class, nodeFeat),
                                classesFeatStd(N_class, nodeFeat));

                    GaussianGenerator generator(rng, gaussian_dist);

                    nodeFeatures(nodeFeat) = generator();
                }

                // Create the node and fill the needed variables

                CNodePtr nodePtr ( new CNode() );
                string label("object");
                nodePtr->setLabel( label );
                nodePtr->setType( nodeType1Ptr );
                nodePtr->setFeatures( nodeFeatures );

                // Add node to the graph
                graph.addNode( nodePtr );

                // Set the ground truth
                groundTruth[ nodePtr->getID() ] = N_class;

                nodes.push_back( nodePtr );

            }
        }

        size_t N_nodes = graph.getNodes().size();

        //
        // Create the edges (relations between the nodes)
        //
        for ( size_t node1 = 0; node1 < N_nodes - 1; node1++ )
        {
            for ( size_t node2 = node1+1; node2 < N_nodes; node2++ )
            {
                // Decide if these two nodes are linked
                size_t random = rand() % 100;

                if ( relationsFrequency( groundTruth[nodes[node1]->getID()],
                                         groundTruth[nodes[node2]->getID()] ) > random )
                {
                    // Generate the edge features
                    VectorXd edgeFeatures( N_edgeFeatures );

                    // Diference of centroid heights
                    edgeFeatures(0) = std::abs( nodes[node1]->getFeatures()(0) -
                                                nodes[node2]->getFeatures()(0) );
                    // Sizes ratio
                    if ( nodes[node1]->getFeatures()(1) > nodes[node2]->getFeatures()(1) )
                        edgeFeatures(1) = nodes[node2]->getFeatures()(1) /
                                                nodes[node1]->getFeatures()(1);
                    else
                        edgeFeatures(1) = nodes[node1]->getFeatures()(1) /
                                                nodes[node2]->getFeatures()(1);

                    // node1 on_top_of node2?
                    edgeFeatures(2) = on_top_of( groundTruth[nodes[node1]->getID()],
                                                 groundTruth[nodes[node2]->getID()] );

                    // node2 on_top_of node1?
                    edgeFeatures(3) = on_top_of( groundTruth[nodes[node2]->getID()],
                                                 groundTruth[nodes[node1]->getID()] );

                    // Bias
                    edgeFeatures(4) = 1;

                    CEdgePtr edgePtr( new CEdge( nodes[node1],
                                                 nodes[node2],
                                                 edgeType1Ptr,
                                                 edgeFeatures ) );

                    graph.addEdge( edgePtr );
                }

            }
        }

        // Add the graph and the ground truth to the tratining data set
        trainingDataset.addGraph( graph );
        trainingDataset.addGraphGroundTruth( groundTruth );
    }

/*------------------------------------------------------------------------------
 *
 *                               TRAINING
 *
 *----------------------------------------------------------------------------*/

    cout << "---------------------------------------------" << endl;
    cout << "                 TRAINING " << endl;
    cout << "---------------------------------------------" << endl << endl;

    //
    // Set the training options
    //
    TTrainingOptions trainingOptions;
    trainingOptions.showTrainingProgress = true;
    trainingOptions.showTrainedWeights   = true;
    trainingOptions.l2Regularization     = true;
    trainingOptions.nodeLambda           = 5;
    trainingOptions.edgeLambda           = 100;

    //trainingOptions.trainingType = "decoding";
    //trainingOptions.trainingType = "inference";

    trainingDataset.setTrainingOptions( trainingOptions );

    //
    // Let's go training!
    //
    trainingDataset.train();

/*------------------------------------------------------------------------------
 *
 *                      CHECKING THE PGM PERFORMANCE
 *
 *----------------------------------------------------------------------------*/


    double totalSuccess_Greedy  = 0;
    double totalSuccess_ICM     = 0;
    double totalSuccess_Exact   = 0;
    double totalSuccess_LBP     = 0;
    double totalSuccess_TRPBP   = 0;
    double totalSuccess_MaxNodePot     = 0;
    double totalSuccess_AlphaExpansion = 0;
    double totalSuccess_AlphaBetaSwap  = 0;

    double totalNumberOfNodes   = 0;

    //
    // Create the scenarios!
    //
    for ( size_t N_test = 0; N_test < 150; N_test++ )
    {
        // Create the graph for the scenario
        CGraph graph;
        std::map<size_t,size_t> groundTruth;

        //
        // Create the nodes of the graph (objects in the scenario)
        //

        vector<CNodePtr> nodes;
        map<size_t,size_t> classesConsidered;

        for ( int count = N_classes-1; count >= 0; count-- )
        {
            // Decide the class of the next object
            size_t N_class;
            size_t classToCheck = rand() % N_classes;

            while ( classesConsidered[classToCheck] == 1 )
                classToCheck = rand() % N_classes;

            N_class = classToCheck;
            classesConsidered[N_class] = 1;

            // Decide which objects appear in the scenario
            size_t random = rand() % 100;

            if ( frequencyOfOcurrence(N_class) > random )
            {
                // Generate node features
                VectorXd nodeFeatures(N_nodeFeatures);

                for ( size_t nodeFeat = 0; nodeFeat < N_nodeFeatures; nodeFeat++ )
                {
                    NormalDistribution gaussian_dist(
                                classesFeatMean(N_class, nodeFeat),
                                classesFeatStd(N_class, nodeFeat));

                    GaussianGenerator generator(rng, gaussian_dist);

                    nodeFeatures(nodeFeat) = generator();
                }

                // Create the node and fill the needed variables

                CNodePtr nodePtr ( new CNode() );
                string label("object");
                nodePtr->setLabel( label );
                nodePtr->setType( nodeType1Ptr );
                nodePtr->setFeatures( nodeFeatures );

                // Add node to the graph
                graph.addNode( nodePtr );

                // Set the ground truth
                groundTruth[ nodePtr->getID() ] = N_class;

                nodes.push_back( nodePtr );

            }
        }

        size_t N_nodes = graph.getNodes().size();

        //
        // Create the edges (relations between the nodes)
        //

        for ( size_t node1 = 0; node1 < N_nodes - 1; node1++ )
        {
            for ( size_t node2 = node1+1; node2 < N_nodes; node2++ )
            {
                // Decide if these two nodes are linked
                size_t random = rand() % 100;


                if ( relationsFrequency( groundTruth[nodes[node1]->getID()],
                                         groundTruth[nodes[node2]->getID()] ) > random )
                {
                    // Generate the edge features
                    VectorXd edgeFeatures( N_edgeFeatures );

                    // Diference of centroid heights
                    edgeFeatures(0) = std::abs( nodes[node1]->getFeatures()(0) -
                                                nodes[node2]->getFeatures()(0) );
                    // Sizes ratio
                    if ( nodes[node1]->getFeatures()(1) > nodes[node2]->getFeatures()(1) )
                        edgeFeatures(1) = nodes[node2]->getFeatures()(1) /
                                nodes[node1]->getFeatures()(1);
                    else
                        edgeFeatures(1) = nodes[node1]->getFeatures()(1) /
                                nodes[node2]->getFeatures()(1);

                    // node1 on_top_of node2?
                    edgeFeatures(2) = on_top_of( groundTruth[nodes[node1]->getID()],
                            groundTruth[nodes[node2]->getID()] );

                    // node2 on_top_of node1?
                    edgeFeatures(3) = on_top_of( groundTruth[nodes[node2]->getID()],
                            groundTruth[nodes[node1]->getID()] );

                    // Bias
                    edgeFeatures(4) = 1;

                    CEdgePtr edgePtr( new CEdge( nodes[node1],
                                                 nodes[node2],
                                                 edgeType1Ptr,
                                                 edgeFeatures ) );

                    graph.addEdge( edgePtr );
                }

            }
        }

        totalNumberOfNodes += graph.getNodes().size();

        graph.computePotentials();


        CDecodeICM decodeICM;
        CDecodeICMGreedy decodeICMGreedy;
        CDecodeExact decodeExact;
        CDecodeLBP decodeLBP;
        CDecodeTRPBP decodeTRPBP;
        CDecodeAlphaExpansion decodeAlphaExpansion;
        CDecodeAlphaExpansion decodeAlphaBetaSwap;
        CDecodeMaxNodePot decodeMaxNodePot;

        TInferenceOptions options;
        options.maxIterations = 100;
        options.initialAssignation = "Random";

        std::map<size_t,size_t> resultsMap;
        std::map<size_t,size_t>::iterator it;
        double success;

        //
        // MaxNodePot
        //

        decodeMaxNodePot.setOptions( options );
        decodeMaxNodePot.decode( graph, resultsMap );

        success = 0;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_MaxNodePot++;
                success++;
            }
        }

        //
        // Greedy
        //

        decodeICMGreedy.setOptions( options );
        decodeICMGreedy.decode( graph, resultsMap );

        success = 0;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            //cout << "Node id " << it->first << " labeled as " << it->second <<
            //        " being " << groundTruth[ it->first ] << endl;
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_Greedy++;
                success++;
            }
        }


        //
        // ICM
        //

        decodeICM.setOptions( options );
        decodeICM.decode( graph, resultsMap );

        success = 0;

        //cout << "ICM" << endl;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_ICM++;
                success++;
            }
        }

        //
        // EXACT
        //

        decodeExact.setOptions( options );
        decodeExact.decode( graph, resultsMap );

        success = 0;

        //cout << "Exact" << endl;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_Exact++;
                success++;
            }
        }


        //
        // LBP
        //

        options.convergency = 0.0001;
        options.maxIterations = 10;
        options.particularD["smoothing"] = 0.9; // A number between 0 and 1

        decodeLBP.setOptions( options );
        decodeLBP.decode( graph, resultsMap );

        success = 0;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_LBP++;
                success++;
            }
        }

        options.convergency = 0.0001;
        options.maxIterations = 10;
        //options.particularD["smoothing"] = 0.9; // A number between 0 and 1

        decodeTRPBP.setOptions( options );
        decodeTRPBP.decode( graph, resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruth[ it->first ])
                totalSuccess_TRPBP++;

        //
        // ALPHA-EXPANSION
        //

        options.maxIterations = 10000;

        decodeAlphaExpansion.setOptions( options );
        decodeAlphaExpansion.decode( graph, resultsMap );

        success = 0;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_AlphaExpansion++;
                success++;
            }
        }

        //
        // ALPHA-BETA-SWAp
        //

        options.maxIterations = 100;

        decodeAlphaBetaSwap.setOptions( options );
        decodeAlphaBetaSwap.decode( graph, resultsMap );

        success = 0;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_AlphaBetaSwap++;
                success++;
            }
        }

    }

    cout << endl;
    cout << "---------------------------------------------" << endl;
    cout << "              PGM PERFORMANCE " << endl;
    cout << "---------------------------------------------" << endl << endl;

    cout << "Total MaxNodePot success: " << 100*(totalSuccess_MaxNodePot / totalNumberOfNodes) << "%" << endl;
    cout << "Total Greedy     success: " << 100*(totalSuccess_Greedy / totalNumberOfNodes) << "%" << endl;
    cout << "Total ICM        success: " << 100*(totalSuccess_ICM / totalNumberOfNodes) << "%" << endl;
    cout << "Total Exact      success: " << 100*(totalSuccess_Exact / totalNumberOfNodes) << "%" << endl;
    cout << "Total LBP        success: " << 100*(totalSuccess_LBP / totalNumberOfNodes) << "%" << endl;
    cout << "Total TRPBP      success: " << 100*(totalSuccess_TRPBP / totalNumberOfNodes) << "%" << endl;
    cout << "Total AlphaExpan success: " << 100*(totalSuccess_AlphaExpansion / totalNumberOfNodes) << "%" << endl;
    cout << "Total AlphaBetaS success: " << 100*(totalSuccess_AlphaBetaSwap / totalNumberOfNodes) << "%" << endl;

    cout << endl;


    return 1;
}

