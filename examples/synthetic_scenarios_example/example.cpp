
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

typedef boost::normal_distribution<double> NormalDistribution;
typedef boost::mt19937 RandomGenerator;
typedef boost::variate_generator<RandomGenerator&, \
                        NormalDistribution> GaussianGenerator;

int main (int argc, char* argv[])
{

    // Initiate Random Number generator with current time
    static RandomGenerator rng(static_cast<unsigned> (time(0)));

    CTrainingDataSet trainingDataset;

    /* Classes:
     *  0: bin
     *  1: table
     *  2: book
     */

    size_t N_classes = 3;

    /* Node features:
     *  0: centroid height
     *  1: size
     *  2: bias
     */
    size_t N_nodeFeatures = 3;

    MatrixXd classesFeatMean(N_classes,N_nodeFeatures);
    classesFeatMean <<  0.15,   0.3,      1,
                        0.75,   1,        1,
                        0.85,   0.20,     1;

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
    CEdgeTypePtr edgeType1Ptr ( new CEdgeType(N_edgeFeatures, N_classes, N_classes) );

    label = "Edge between two objects";
    edgeType1Ptr->setLabel( label );
    //trainingDataset.addEdgeType( edgeType1Ptr );

    VectorXi typeOfEdgeFeatures( N_edgeFeatures );
    typeOfEdgeFeatures << 0, 0, 1, 1, 0;

    trainingDataset.addEdgeType( edgeType1Ptr, typeOfEdgeFeatures );

    // Number of scenarios
    size_t N_syntheticTrainingSamples = 2000;

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
            size_t classToCheck = rand() % 3;

            while ( classesConsidered[classToCheck] == 1 )
                classToCheck = rand() % 3;

            N_class = classToCheck;
            classesConsidered[N_class] = 1;

            //cout << "Object class" << N_class << endl;

            // Decide if the object of this class appear in the scenario
            size_t random = rand() % 100;
            //cout << "Random: "<< random << endl;

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

                    //cout << "Object class " << N_class << " feature " << nodeFeat << " value " << generator() << endl;

                    nodeFeatures(nodeFeat) = generator();
                }

                // Create the node and fill the needed variables

                CNodePtr nodePtr ( new CNode() );
                string label("object");
                nodePtr->setLabel( label );
                nodePtr->setType( nodeType1Ptr );
                nodePtr->setFeatures( nodeFeatures );

                //cout << "Node" << endl << *nodePtr << endl;

                // Add node to the graph
                graph.addNode( nodePtr );

                // Set the ground truth
                groundTruth[ nodePtr->getId() ] = N_class;

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
                //cout << "Testing <" << node1 << "," << node2 << ">" << endl;
                // Decide if these two nodes are linked
                size_t random = rand() % 100;

                /*cout << "Testing " << random << "lower than " <<
                        relationsFrequency( groundTruth[nodes[node1]->getId()], groundTruth[nodes[node2]->getId()] ) <<
                        " GT 1: " << groundTruth[nodes[node1]->getId()] << " GT 2: " << groundTruth[nodes[node2]->getId()] <<
                        endl;*/

                if ( relationsFrequency( groundTruth[nodes[node1]->getId()],
                                         groundTruth[nodes[node2]->getId()] ) > random )
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
                    edgeFeatures(2) = on_top_of( groundTruth[nodes[node1]->getId()], groundTruth[nodes[node2]->getId()] );

                    // node2 on_top_of node1?
                    edgeFeatures(3) = on_top_of( groundTruth[nodes[node2]->getId()], groundTruth[nodes[node1]->getId()] );

                    // Bias
                    edgeFeatures(4) = 1;

                    //cout << "Node of type <" << groundTruth[nodes[node1]->getId()] << "," << groundTruth[nodes[node2]->getId()] << ">" << " ID <" << nodes[node1]->getId() << "," << nodes[node2]->getId() << ">" << endl;
                    //cout << "Edge features" << edgeFeatures << endl;

                    CEdgePtr edgePtr( new CEdge( nodes[node1], nodes[node2], edgeType1Ptr) );
                    edgePtr->setFeatures( edgeFeatures );

                    graph.addEdge( edgePtr );
                }

            }
        }

        // Add the graph and the ground truth to the tratining data set
        trainingDataset.addGraph( graph );
        trainingDataset.addGraphGroundTruth( groundTruth );
    }

    // Let's go training!
    trainingDataset.train();



    // Let's go testing!

    double totalSuccess_Greedy = 0;
    double totalSuccess_ICM = 0;
    double totalSuccess_Exact = 0;
    double totalNumberOfNodes = 0;

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
            size_t classToCheck = rand() % 3;

            while ( classesConsidered[classToCheck] == 1 )
                classToCheck = rand() % 3;

            N_class = classToCheck;
            classesConsidered[N_class] = 1;

            // Decide which objects appear in the scenario
            size_t random = rand() % 100;
            //cout << "Random: "<< random << endl;

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

                    //cout << "Object class " << N_class << " feature " << nodeFeat << " value " << generator() << endl;

                    nodeFeatures(nodeFeat) = generator();
                }

                // Create the node and fill the needed variables

                CNodePtr nodePtr ( new CNode() );
                string label("object");
                nodePtr->setLabel( label );
                nodePtr->setType( nodeType1Ptr );
                nodePtr->setFeatures( nodeFeatures );

                //cout << "Node" << endl << *nodePtr << endl;

                // Add node to the graph
                graph.addNode( nodePtr );

                // Set the ground truth
                groundTruth[ nodePtr->getId() ] = N_class;

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
                //cout << "Testing <" << node1 << "," << node2 << ">" << endl;
                // Decide if these two nodes are linked
                size_t random = rand() % 100;

                /*cout << "Testing " << random << "lower than " <<
                        relationsFrequency( groundTruth[nodes[node1]->getId()], groundTruth[nodes[node2]->getId()] ) <<
                        " GT 1: " << groundTruth[nodes[node1]->getId()] << " GT 2: " << groundTruth[nodes[node2]->getId()] <<
                        endl;*/

                if ( relationsFrequency( groundTruth[nodes[node1]->getId()],
                                         groundTruth[nodes[node2]->getId()] ) > random )
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
                    edgeFeatures(2) = on_top_of( groundTruth[nodes[node1]->getId()],
                                                 groundTruth[nodes[node2]->getId()] );

                    // node2 on_top_of node1?
                    edgeFeatures(3) = on_top_of( groundTruth[nodes[node2]->getId()],
                                                 groundTruth[nodes[node1]->getId()] );

                    // Bias
                    edgeFeatures(4) = 1;

                    //cout << "Edge features" << edgeFeatures << endl;

                    CEdgePtr edgePtr( new CEdge( nodes[node1], nodes[node2], edgeType1Ptr) );
                    edgePtr->setFeatures( edgeFeatures );

                    graph.addEdge( edgePtr );
                }

            }
        }

        totalNumberOfNodes += graph.getNodes().size();

        graph.computePotentials();

        cout << "------------------------------------------------------" << endl;
        cout << "Graph" << endl << graph << endl;

        TOptions options;
        options.maxIterations = 100;

        std::map<size_t,size_t> resultsMap;
        std::map<size_t,size_t>::iterator it;

        //
        // Greedy
        //
        decodeICMGreedy( graph, options, resultsMap );

        double success = 0;

        //cout << "ICM Greedy" << endl;

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

        //cout << " Success = " << success/resultsMap.size()*100 << "% " << endl;

        //cout << "-------------------------------------------------" << endl;

        //
        // ICM
        //
        decodeICM( graph, options, resultsMap );

        success = 0;

        //cout << "ICM" << endl;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            //cout << "Node id " << it->first << " labeled as " << it->second <<
            //        " being " << groundTruth[ it->first ] << endl;
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_ICM++;
                success++;
            }
        }

        //cout << " Success = " << success/resultsMap.size()*100 << "% " << endl;

        //cout << "-------------------------------------------------" << endl;

        //
        // EXACT
        //
        decodeExact( graph, options, resultsMap );

        success = 0;

        cout << "Exact" << endl;

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            cout << "Node id " << it->first << " labeled as " << it->second <<
                    " being " << groundTruth[ it->first ] << endl;
            if ( it->second == groundTruth[ it->first ])
            {
                totalSuccess_Exact++;
                success++;
            }
        }

        //cout << " Success = " << success/resultsMap.size()*100 << "% " << endl;

        cout << "-------------------------------------------------" << endl;
    }

    cout << "Total Greedy success: " << 100*(totalSuccess_Greedy / totalNumberOfNodes) << "%" << endl;
    cout << "Total ICM    success: " << 100*(totalSuccess_ICM / totalNumberOfNodes) << "%" << endl;
    cout << "Total Exact  success: " << 100*(totalSuccess_Exact / totalNumberOfNodes) << "%" << endl;


    return 1;
}

