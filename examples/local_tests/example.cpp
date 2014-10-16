
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
#include "inference.hpp"

#include <iostream>
#include <math.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <ctime>

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


CTrainingDataSet trainingDataset;


void fillGraph( CGraph &graph,
                const Eigen::MatrixXd &graph_nodeFeatures,
                CNodeTypePtr nodeType,
                const Eigen::MatrixXi &graph_adj,
                CEdgeTypePtr edgeType,
                vector<Eigen::MatrixXi> &g1_relations,
                const Eigen::VectorXi &g1_groundTruth,
                std::map<size_t,size_t> &groundTruth // returned GT taking into account the node IDs
              )
{


    // 1. Add nodes to the graph
    // 2. Add edges

    size_t n_nodes = graph_nodeFeatures.rows();
    size_t n_features = graph_nodeFeatures.cols();

    vector<CNodePtr> nodes;

    //
    // ADD NODES
    //

    for ( size_t i = 0; i < n_nodes; i++ )
    {
        Eigen::VectorXd node_feat(n_features);
        node_feat = graph_nodeFeatures.row(i);

        Eigen::VectorXd multipliers(5);
        multipliers << 90, 100, 50, 40, 50;
        //multipliers << 50, 60, 50, 50, 50;
        //node_feat = node_feat.cwiseProduct( multipliers );

        CNodePtr nodePtr ( new CNode( nodeType, node_feat ) );
        nodes.push_back( nodePtr );

        graph.addNode( nodePtr );

        groundTruth[ nodePtr->getID() ] = g1_groundTruth(i);

    }   

    //
    // ADD EDGES
    //

    for ( size_t row = 0; row < n_nodes; row++ )
    {
        for ( size_t col = row; col < n_nodes; col++ )
        {
            if ( graph_adj(row,col) == 1 )
            {
                // Retrieve the nodes linked by the edge and their features
                CNodePtr node1 = nodes.at(row);
                CNodePtr node2 = nodes.at(col);

                Eigen::VectorXd &feat1 = node1->getFeatures();
                Eigen::VectorXd &feat2 = node2->getFeatures();

                // Compute edge features
                Eigen::VectorXd edgeFeatures( edgeType->getWeights().size());

                // Perpendicularity

                edgeFeatures(0) = (std::abs(feat1(0) - feat2(0))==0) ? 0 : 1;

                // Height distance between centers
                edgeFeatures(1) = std::abs((float)feat1(1) - (float)feat2(1));

                // Ratio between areas
                float a1 = feat1(2);
                float a2 = feat2(2);

                if ( a1 < a2 )
                {
                    float aux = a2;
                    a2 = a1;
                    a1 = aux;
                }

                //edgeFeatures(2) = a1 / a2;

                //  Difference between elongations
                //edgeFeatures(3) = std::abs(feat1(3) - feat2(3));

                // Use the isOn semantic relation
                //edgeFeatures(4) = (g1_relations[0])(row,col);
                edgeFeatures(2) = (g1_relations[0])(row,col);

                // Coplanar semantic relation
                //edgeFeatures(5) = (g1_relations[1])(row,col);

                // Bias feature
                //edgeFeatures(6) = 1;
                edgeFeatures(3) = 1;

                //Eigen::VectorXd multipliers(7);
                //multipliers << 0, 0, 0, 0, 0, 0, 1;
                //multipliers << 0, 0, 0, 0, 0, 0, 0;

                Eigen::VectorXd multipliers(4);
                multipliers << 1, 1, 10, 1;

                //edgeFeatures = edgeFeatures.cwiseProduct( multipliers );

                CEdgePtr edgePtr ( new CEdge( node1, node2,
                                            edgeType, edgeFeatures ) );
                graph.addEdge( edgePtr );
            }
        }
    }

}

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

    size_t N_classes_type_1        = 11;
    size_t N_nf   = 5;
    size_t N_ef   = 4;

    // OBJECTS

    CNodeTypePtr simpleNodeType1Ptr( new CNodeType(N_classes_type_1,
                                                   N_nf,
                                                   string("Objects") ) );

    CEdgeTypePtr simpleEdgeType1Ptr ( new CEdgeType(N_ef,
                                                    simpleNodeType1Ptr,
                                                    simpleNodeType1Ptr,
                                                    string("Edges between two objects")) );


    trainingDataset.addNodeType( simpleNodeType1Ptr );
    trainingDataset.addEdgeType( simpleEdgeType1Ptr );

    //
    // Create a set of training graphs
    //

    vector<CGraph>                      graphs;
    vector<std::map<size_t,size_t> >    groundTruths;

/*--------------------------------- Graph 1 ----------------------------------*/


    cout << "Graph 1" << endl;

    CGraph graph1;
    std::map<size_t,size_t> groundTruth1;

    Eigen::MatrixXd g1_nf(11,5);

    g1_nf << 1, 0.725256, 0.347833, 2.32114, 1,
             0, 0, 1.46145, 2.86637, 1,
             1, 0.513579, 0.398025, 1.85308, 1,
             0, 0.896507, 0.969337, 1.7035, 1,
             0, 0.748331, 0.585186, 1.47532, 1,
             1, 0.963636, 0.125714, 1, 1,
             1, 0.93444, 0.212592, 1.52702, 1,
             1, 1.05757, 0.236803, 3.22358, 1,
             0, 0.398408, 0.131399, 1, 1,
             0, 0.0300602, 0.129521, 2.03183, 1,
             1, 0.469344, 0.505431, 1.45506, 1;

    Eigen::MatrixXi g1_adj(11,11);

    g1_adj <<   0,  1,  1,  1,  1,  1,  0,  0,  1,  1,  0,
                1,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,
                1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                1,  0,  0,  0,  1,  1,  1,  1,  1,  0,  1,
                1,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,
                1,  0,  0,  1,  1,  0,  1,  0,  0,  0,  0,
                0,  0,  0,  1,  1,  1,  0,  1,  1,  0,  1,
                0,  0,  0,  1,  1,  0,  1,  0,  0,  0,  1,
                1,  1,  0,  1,  1,  0,  1,  0,  0,  1,  0,
                1,  1,  0,  0,  1,  0,  0,  0,  1,  0,  0,
                0,  0,  0,  1,  1,  0,  1,  1,  0,  0,  0;

    Eigen::MatrixXi isOn_relation(11,11);

    isOn_relation.setZero();
    isOn_relation(3,5) = 1;
    isOn_relation(4,5) = 1;

    Eigen::MatrixXi coplanar_relation(11,11);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g1_relations;

    g1_relations.push_back( isOn_relation );
    g1_relations.push_back( coplanar_relation );

    Eigen::VectorXi g1_groundTruth(11);

    g1_groundTruth << 4, 0, 1, 2, 2, 6, 1, 1, 5, 0, 3;

    fillGraph( graph1, g1_nf, simpleNodeType1Ptr,
                  g1_adj, simpleEdgeType1Ptr, g1_relations,
                  g1_groundTruth, groundTruth1 );

    graphs.push_back( graph1 );
    groundTruths.push_back( groundTruth1 );

    cout << graph1 << endl;

/*--------------------------------- Graph 2 ----------------------------------*/

    cout << "Graph 2" << endl;

    CGraph graph2;
    std::map<size_t,size_t> groundTruth2;

    Eigen::MatrixXd g2_nf(5,5);

    g2_nf << 0, 0.745944,   1.20098,    1.68726,    1,
             1, 0.523566,   0.255477,   1.72744,    1,
             0, 0,          1.49268,    2.54707,    1,
             1, 0.359525,   0.268375,   2.8157,     1,
             0, 0.446692,   0.129895,   1,          1;

    Eigen::MatrixXi g2_adj(5,5);

    g2_adj << 0,  1,  0,  1,  1,
            1,  0,  1,  1,  0,
            0,  1,  0,  1,  1,
            1,  1,  1,  0,  1,
            1,  0,  1,  1,  0;

    isOn_relation.resize(5,5);

    isOn_relation.setZero();

    coplanar_relation.resize(5,5);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g2_relations;

    g2_relations.push_back( isOn_relation );
    g2_relations.push_back( coplanar_relation );

    Eigen::VectorXi g2_groundTruth(5);

    g2_groundTruth << 2, 3, 0, 3, 5;

    fillGraph( graph2, g2_nf, simpleNodeType1Ptr,
               g2_adj, simpleEdgeType1Ptr, g2_relations,
               g2_groundTruth, groundTruth2 );

    graphs.push_back( graph2 );
    groundTruths.push_back( groundTruth2 );


///*--------------------------------- Graph 3 ----------------------------------*/

    cout << "Graph 3" << endl;

    CGraph graph3;
    std::map<size_t,size_t> groundTruth3;

    Eigen::MatrixXd g3_nf(10,5);

    g3_nf << 1, 1.56711,    3.42284,    2.00889, 1,
             1, 1.02852,    0.339804,   1.54119, 1,
             1, 1.02507,    0.317727,   1.77383, 1,
             0, 0.880216,   1.45163,    2.04593, 1,
             1, 1.04219,    0.229974,   1.77708, 1,
             1, 1.66558,    0.171681,   2.3654,  1,
             0, 0.488131,   0.233171,   1.51503, 1,
             1, 0.699061,   0.120062,   1.55154, 1,
             0, 0,          2.84036,    1.99605, 1,
             1, 0.167627,   0.352816,   3.80501, 1;

    Eigen::MatrixXi g3_adj(10,10);

    g3_adj <<   0,  1,  1,  1,  1,  1,  0,  0,  0,  1,
            1,  0,  1,  1,  0,  0,  0,  0,  0,  0,
            1,  1,  0,  1,  1,  1,  1,  1,  0,  1,
            1,  1,  1,  0,  1,  1,  1,  1,  0,  1,
            1,  0,  1,  1,  0,  1,  1,  1,  0,  1,
            1,  0,  1,  1,  1,  0,  0,  0,  0,  0,
            0,  0,  1,  1,  1,  0,  0,  1,  1,  0,
            0,  0,  1,  1,  1,  0,  1,  0,  1,  0,
            0,  0,  0,  0,  0,  0,  1,  1,  0,  1,
            1,  0,  1,  1,  1,  0,  0,  0,  1,  0;

    isOn_relation.resize(10,10);
    isOn_relation.setZero();
    isOn_relation(1,3) = 1;
    isOn_relation(2,3) = 1;
    isOn_relation(3,4) = 1;

    coplanar_relation.resize(10,10);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g3_relations;

    g3_relations.push_back( isOn_relation );
    g3_relations.push_back( coplanar_relation );

    Eigen::VectorXi g3_groundTruth(10);

    g3_groundTruth << 1, 6, 6, 2, 9, 10, 5, 4, 0, 1;

    fillGraph( graph3, g3_nf, simpleNodeType1Ptr,
                  g3_adj, simpleEdgeType1Ptr, g3_relations,
                  g3_groundTruth, groundTruth3 );

    graphs.push_back( graph3 );
    groundTruths.push_back( groundTruth3 );

///*--------------------------------- Graph 4 ----------------------------------*/

    cout << "Graph 4" << endl;

    CGraph graph4;
    std::map<size_t,size_t> groundTruth4;

    Eigen::MatrixXd g4_nf(7,5);

    g4_nf << 1, 1.58673,    2.5301,     1.79962, 1,
             1, 1.01892,    0.198802,   1.46393, 1,
             1, 1.00455,    0.230172,   1.67413, 1,
             0, 0.744138,   0.89911,    1.57856, 1,
             0, 0.477457,   0.187906,   1.87622, 1,
             1, 0.7026,     0.113952,   1.44738, 1,
             0, 0,          2.75497,    1.87592, 1;

    Eigen::MatrixXi g4_adj(7,7);

    g4_adj <<   0,  1,  1,  1,  0,  0,  0,
                1,  0,  1,  1,  1,  1,  0,
                1,  1,  0,  1,  1,  1,  0,
                1,  1,  1,  0,  1,  1,  0,
                0,  1,  1,  1,  0,  1,  1,
                0,  1,  1,  1,  1,  0,  1,
                0,  0,  0,  0,  1,  1,  0;

    isOn_relation.resize(7,7);
    isOn_relation.setZero();
    isOn_relation(1,3) = 1;
    isOn_relation(2,3) = 1;

    coplanar_relation.resize(7,7);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g4_relations;

    g4_relations.push_back( isOn_relation );
    g4_relations.push_back( coplanar_relation );

    Eigen::VectorXi g4_groundTruth(7);

    g4_groundTruth << 1, 6, 9, 2, 5, 4, 0;

    fillGraph( graph4, g4_nf, simpleNodeType1Ptr,
                  g4_adj, simpleEdgeType1Ptr, g4_relations,
                  g4_groundTruth, groundTruth4 );

    graphs.push_back( graph4 );
    groundTruths.push_back( groundTruth4 );

///*--------------------------------- Graph 5 ----------------------------------*/

    cout << "Graph 5" << endl;

    // Desktop 6
    CGraph graph5;
    std::map<size_t,size_t> groundTruth5;

    Eigen::MatrixXd g5_nf(8,5);

    g5_nf << 1, 1.10316,    0.363647,   1.61694,    1,
             1, 1.00384,    0.23474,    2.7587,     1,
             1, 0.988903,   0.118134,   1,          1,
             0, 0.789307,   1.62451,    2.07762,    1,
             1, 0.672498,   1.8316,     1.78692,    1,
             1, 0.976089,   0.244898,   1,          1,
             0, 0.502879,   0.186037,   2.29688,    1,
             0, 0,          2.50486,    1.43262,    1;

    Eigen::MatrixXi g5_adj(8,8);

    g5_adj <<   0,  1,  1,  1,  1,  1,  0,  0,
                1,  0,  1,  1,  1,  0,  0,  0,
                1,  1,  0,  1,  0,  0,  0,  0,
                1,  1,  1,  0,  1,  1,  1,  0,
                1,  1,  0,  1,  0,  1,  1,  1,
                1,  0,  0,  1,  1,  0,  1,  0,
                0,  0,  0,  1,  1,  1,  0,  0,
                0,  0,  0,  0,  1,  0,  0,  0;

    isOn_relation.resize(8,8);
    isOn_relation.setZero();
    isOn_relation(0,3) = 1;
    isOn_relation(2,3) = 1;

    coplanar_relation.resize(8,8);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g5_relations;

    g5_relations.push_back( isOn_relation );
    g5_relations.push_back( coplanar_relation );

    Eigen::VectorXi g5_groundTruth(8);

    g5_groundTruth << 6, 1, 8, 2, 1, 1, 5, 0;

    fillGraph( graph5, g5_nf, simpleNodeType1Ptr,
                  g5_adj, simpleEdgeType1Ptr, g5_relations,
                  g5_groundTruth, groundTruth5 );

    graphs.push_back( graph5 );
    groundTruths.push_back( groundTruth5 );


///*--------------------------------- Graph 6 ----------------------------------*/

    cout << "Graph 6" << endl;

    // Desktop 7
    CGraph graph6;
    std::map<size_t,size_t> groundTruth6;

    Eigen::MatrixXd g6_nf(7,5);

    g6_nf << 1, 0.975437,   1.21382,    2.28555,    1,
             1, 0.951433,   0.270196,   2.29035,    1,
             0, 0.766445,   1.00624,    1.6777,     1,
             0, 0,          1.13219,    1.64994,    1,
             1, 0.985712,   0.208164,   1.67471,    1,
             0, 0.499913,   0.208579,   1.95149,    1,
             1, 0.970966,   0.16066,    1,          1;

    Eigen::MatrixXi g6_adj(7,7);

    g6_adj <<   0,  1,  1,  1,  0,  0,  1,
                1,  0,  1,  0,  1,  1,  1,
                1,  1,  0,  0,  1,  1,  1,
                1,  0,  0,  0,  0,  1,  0,
                0,  1,  1,  0,  0,  1,  0,
                0,  1,  1,  1,  1,  0,  0,
                1,  1,  1,  0,  0,  0,  0;

    isOn_relation.resize(7,7);
    isOn_relation.setZero();
    isOn_relation(2,6) = 1;

    coplanar_relation.resize(7,7);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g6_relations;

    g6_relations.push_back( isOn_relation );
    g6_relations.push_back( coplanar_relation );

    Eigen::VectorXi g6_groundTruth(7);

    g6_groundTruth << 1, 1, 2, 0, 4, 5, 6;

    fillGraph( graph6, g6_nf, simpleNodeType1Ptr,
                  g6_adj, simpleEdgeType1Ptr, g6_relations,
                  g6_groundTruth, groundTruth6 );

    graphs.push_back( graph6 );
    groundTruths.push_back( groundTruth6 );


///*--------------------------------- Graph 7 ----------------------------------*/

    cout << "Graph 7" << endl;

    // Desktop 8
    CGraph graph7;
    std::map<size_t,size_t> groundTruth7;

    Eigen::MatrixXd g7_nf(7,5);

    g7_nf << 1, 1.08949,    0.44,       1.51331,    1,
             1, 1.07257,    0.144868,   1,          1,
             0, 0.802159,   1.19101,    1.78961,    1,
             1, 0.728912,   0.128203,   2.15983,    1,
             0, 0,          3.7626,     2.86856,    1,
             0, 0.0142013,  0.175163,   2.24823,    1,
             0, 0.540925,   0.205526,   1,          1;

    Eigen::MatrixXi g7_adj(7,7);

    g7_adj << 0,  1,  1,  1,  0,  0,  1,
            1,  0,  1,  0,  0,  0,  0,
            1,  1,  0,  1,  0,  0,  1,
            1,  0,  1,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  1,  1,
            0,  0,  0,  0,  1,  0,  1,
            1,  0,  1,  1,  1,  1,  0;

    isOn_relation.resize(7,7);
    isOn_relation.setZero();
    isOn_relation(0,2) = 1;
    isOn_relation(1,2) = 1;

    coplanar_relation.resize(7,7);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g7_relations;

    g7_relations.push_back( isOn_relation );
    g7_relations.push_back( coplanar_relation );

    Eigen::VectorXi g7_groundTruth(7);

    g7_groundTruth << 6, 6, 2, 4, 0, 0, 5;

    fillGraph( graph7, g7_nf, simpleNodeType1Ptr,
                  g7_adj, simpleEdgeType1Ptr, g7_relations,
                  g7_groundTruth, groundTruth7 );

    graphs.push_back( graph7 );
    groundTruths.push_back( groundTruth7 );


///*--------------------------------- Graph 8 ----------------------------------*/


    cout << "Graph 8" << endl;

    // Desktop 9
    CGraph graph8;
    std::map<size_t,size_t> groundTruth8;

    Eigen::MatrixXd g8_nf(5,5);

    g8_nf << 1, 1.0398,     1.45393,    1.85835,    1,
             0, 0.687862,   1.05335,    1.44245,    1,
             0, 0.476001,   0.17945,    1,          1,
             1, 0.150004,   0.163443,   1.72806,    1,
             0, 0,          2.11644,    3.64183,    1;

    Eigen::MatrixXi g8_adj(5,5);

    g8_adj << 0,  1,  0,  0,  0,
            1,  0,  1,  1,  0,
            0,  1,  0,  1,  1,
            0,  1,  1,  0,  1,
            0,  0,  1,  1,  0;

    isOn_relation.resize(5,5);
    isOn_relation.setZero();

    coplanar_relation.resize(5,5);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g8_relations;

    g8_relations.push_back( isOn_relation );
    g8_relations.push_back( coplanar_relation );

    Eigen::VectorXi g8_groundTruth(5);

    g8_groundTruth << 1, 2, 5, 1, 0;

    fillGraph( graph8, g8_nf, simpleNodeType1Ptr,
                  g8_adj, simpleEdgeType1Ptr, g8_relations,
                  g8_groundTruth, groundTruth8 );

    graphs.push_back( graph8 );
    groundTruths.push_back( groundTruth8 );




///*--------------------------------- Graph 9 ----------------------------------*/

    cout << "Graph 9" << endl;

    // Desktop 11
    CGraph graph9;
    std::map<size_t,size_t> groundTruth9;

    Eigen::MatrixXd g9_nf(6,5);

    g9_nf << 1, 1.08032,    0.175311,   1.59692, 1,
             1, 1.04474,    0.153065,   1.75199, 1,
             0, 0.802788,   1.36259,    2.70243, 1,
             1, 0.727563,   0.115821,   1.56757, 1,
             0, 0,          2.24629,    1.73137, 1,
             0, 0.516504,   0.165597,   1,       1;

    Eigen::MatrixXi g9_adj(6,6);

    g9_adj << 0,  1,  1,  1,  0,  0,
             1,  0,  1,  0,  0,  0,
             1,  1,  0,  1,  0,  1,
             1,  0,  1,  0,  1,  1,
             0,  0,  0,  1,  0,  1,
             0,  0,  1,  1,  1,  0;

    isOn_relation.resize(6,6);
    isOn_relation.setZero();
    isOn_relation(0,2) = 1;

    coplanar_relation.resize(6,6);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g9_relations;

    g9_relations.push_back( isOn_relation );
    g9_relations.push_back( coplanar_relation );

    Eigen::VectorXi g9_groundTruth(6);

    g9_groundTruth << 6, 1, 2, 4, 0, 5;

    fillGraph( graph9, g9_nf, simpleNodeType1Ptr,
                  g9_adj, simpleEdgeType1Ptr, g9_relations,
                  g9_groundTruth, groundTruth9 );

    graphs.push_back( graph9 );
    groundTruths.push_back( groundTruth9 );

///*--------------------------------- Graph 10 ----------------------------------*/

    cout << "Graph 10" << endl;

    // Desktop 13
    CGraph graph10;
    std::map<size_t,size_t> groundTruth10;

    Eigen::MatrixXd g10_nf(7,5);

    g10_nf << 1, 1.13981,   0.613381,   1.46317,    1,
              1, 1.01258,   0.167784,   1.47751,    1,
              0, 0.772106,  1.55737,    3.7527,     1,
              1, 1.11485,   0.201305,   2.29086,    1,
              0, 0.487854,  0.161754,   1.53537,    1,
              0, 0,         0.70792,    2.77777,    1,
              0, 0.0221937, 0.310503,   1.72698,    1;

    Eigen::MatrixXi g10_adj(7,7);

    g10_adj << 0,  1,  1,  1,  0,  0,  0,
                1,  0,  1,  1,  0,  0,  0,
                1,  1,  0,  1,  1,  0,  0,
                1,  1,  1,  0,  1,  0,  0,
                0,  0,  1,  1,  0,  1,  1,
                0,  0,  0,  0,  1,  0,  1,
                0,  0,  0,  0,  1,  1,  0;

    isOn_relation.resize(7,7);
    isOn_relation.setZero();
    isOn_relation(1,2) = 1;
    isOn_relation(2,3) = 1;

    coplanar_relation.resize(7,7);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g10_relations;

    g10_relations.push_back( isOn_relation );
    g10_relations.push_back( coplanar_relation );

    Eigen::VectorXi g10_groundTruth(7);

    g10_groundTruth << 1, 6, 2, 8, 5, 0, 0;

    fillGraph( graph10, g10_nf, simpleNodeType1Ptr,
                  g10_adj, simpleEdgeType1Ptr, g10_relations,
                  g10_groundTruth, groundTruth10 );

    graphs.push_back( graph10 );
    groundTruths.push_back( groundTruth10 );


///*--------------------------------- Graph 11 ----------------------------------*/

    cout << "Graph 11" << endl;

    // Desktop 14
    CGraph graph11;
    std::map<size_t,size_t> groundTruth11;

    Eigen::MatrixXd g11_nf(3,5);

    g11_nf << 1, 1.39389,   1.07869,    3.04842, 1,
              0, 0.724976,  0.627546,   3.18692, 1,
              1, 1.02144,   0.168219,   1.53329, 1;

    Eigen::MatrixXi g11_adj(3,3);

    g11_adj << 0,  1,  1,
               1,  0,  1,
               1,  1,  0;

    isOn_relation.resize(3,3);
    isOn_relation.setZero();
    isOn_relation(1,2) = 1;

    coplanar_relation.resize(3,3);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g11_relations;

    g11_relations.push_back( isOn_relation );
    g11_relations.push_back( coplanar_relation );

    Eigen::VectorXi g11_groundTruth(3);

    g11_groundTruth << 1, 2, 6;

    fillGraph( graph11, g11_nf, simpleNodeType1Ptr,
                  g11_adj, simpleEdgeType1Ptr, g11_relations,
                  g11_groundTruth, groundTruth11 );

    graphs.push_back( graph11 );
    groundTruths.push_back( groundTruth11 );


///*--------------------------------- Graph 12 ----------------------------------*/

    cout << "Graph 12" << endl;

    // Desktop 15
    CGraph graph12;
    std::map<size_t,size_t> groundTruth12;

    Eigen::MatrixXd g12_nf(9,5);

    g12_nf << 1, 0.890949,  3.09357,    2.35317, 1,
              0, 0.337975,  0.284948,   1.73898, 1,
              1, 0.722686,  0.397614,   1.45951, 1,
              0, 0.619682,  1.56466,    2.28344, 1,
              1, 0.404655,  0.167949,   1,       1,
              1, 0.726681,  0.198501,   1.52176, 1,
              1, 0.876247,  0.182265,   1.92814, 1,
              1, 0.639947,  0.133658,   2.02595, 1,
              0, 0,         0.36366,    1,       1;

    Eigen::MatrixXi g12_adj(9,9);

    g12_adj << 0,  1,  1,  0,  1,  1,  0,  0,  0,
             1,  0,  1,  1,  1,  0,  0,  0,  0,
             1,  1,  0,  1,  1,  0,  1,  0,  0,
             0,  1,  1,  0,  1,  1,  1,  1,  0,
             1,  1,  1,  1,  0,  0,  1,  0,  0,
             1,  0,  0,  1,  0,  0,  0,  0,  0,
             0,  0,  1,  1,  1,  0,  0,  0,  0,
             0,  0,  0,  1,  0,  0,  0,  0,  1,
             0,  0,  0,  0,  0,  0,  0,  1,  0;

    isOn_relation.resize(9,9);
    isOn_relation.setZero();
    isOn_relation(3,6) = 1;

    coplanar_relation.resize(9,9);

    coplanar_relation.setZero();

    vector<Eigen::MatrixXi> g12_relations;

    g12_relations.push_back( isOn_relation );
    g12_relations.push_back( coplanar_relation );

    Eigen::VectorXi g12_groundTruth(9);

    g12_groundTruth << 1, 5, 4, 2, 3, 1, 6, 4, 0;

    fillGraph( graph12, g12_nf, simpleNodeType1Ptr,
                  g12_adj, simpleEdgeType1Ptr, g12_relations,
                  g12_groundTruth, groundTruth12 );

    graphs.push_back( graph12 );
    groundTruths.push_back( groundTruth12 );


///*--------------------------------- Graph 13 ----------------------------------*/

//    Eigen::MatrixXi g13_adj(11,11);
//, 0,  1,  1,  1,  1,  1,  0,  1,  1,  0
//, 1,  0,  1,  0,  0,  0,  0,  1,  1,  0
//, 1,  1,  0,  0,  0,  0,  0,  0,  0,  0
//, 1,  0,  0,  0,  1,  1,  1,  1,  0,  1
//, 1,  0,  0,  1,  0,  1,  1,  1,  1,  1
//, 1,  0,  0,  1,  1,  0,  1,  0,  0,  0
//, 0,  0,  0,  1,  1,  1,  0,  1,  0,  1
//, 1,  1,  0,  1,  1,  0,  1,  0,  1,  0
//, 1,  1,  0,  0,  1,  0,  0,  1,  0,  0
//, 0,  0,  0,  1,  1,  0,  1,  0,  0,  0

///*--------------------------------- Graph 14 ----------------------------------*/

//    Eigen::MatrixXi g14_adj(11,11);
//, 0,  1,  0,  1,  1,  1
//, 1,  0,  1,  1,  0,  0
//, 0,  1,  0,  1,  1,  0
//, 1,  1,  1,  0,  1,  0
//, 1,  0,  1,  1,  0,  0
//, 1,  0,  0,  0,  0,  0

///*--------------------------------- Graph 15 ----------------------------------*/

//    Eigen::MatrixXi g15_adj(11,11);
//, 0,  1,  1,  1,  0,  0,  0,  1
//, 1,  0,  1,  1,  0,  0,  0,  0
//, 1,  1,  0,  1,  1,  1,  0,  1
//, 1,  1,  1,  0,  1,  1,  0,  1
//, 0,  0,  1,  1,  0,  1,  1,  0
//, 0,  0,  1,  1,  1,  0,  1,  0
//, 0,  0,  0,  0,  1,  1,  0,  1
//, 1,  0,  1,  1,  0,  0,  1,  0

///*--------------------------------- Graph 16 ----------------------------------*/

//    Eigen::MatrixXi g16_adj(11,11);
//, 0,  1,  1,  0,  0,  0
//, 1,  0,  1,  1,  1,  0
//, 1,  1,  0,  1,  1,  0
//, 0,  1,  1,  0,  1,  1
//, 0,  1,  1,  1,  0,  1
//, 0,  0,  0,  1,  1,  0

///*--------------------------------- Graph 17 ----------------------------------*/

//    Eigen::MatrixXi g17_adj(11,11);
//, 0,  1,  1,  1,  1,  0,  0
//, 1,  0,  1,  1,  0,  0,  0
//, 1,  1,  0,  1,  1,  1,  0
//, 1,  1,  1,  0,  1,  1,  1
//, 1,  0,  1,  1,  0,  1,  0
//, 0,  0,  1,  1,  1,  0,  0
//, 0,  0,  0,  1,  0,  0,  0

///*--------------------------------- Graph 18 ----------------------------------*/

//    Eigen::MatrixXi g18_adj(11,11);
//, 0,  1,  1,  1,  0,  0,  1,  1
//, 1,  0,  1,  0,  1,  1,  1,  0
//, 1,  1,  0,  0,  1,  1,  1,  1
//, 1,  0,  0,  0,  0,  1,  0,  0
//, 0,  1,  1,  0,  0,  1,  0,  0
//, 0,  1,  1,  1,  1,  0,  0,  0
//, 1,  1,  1,  0,  0,  0,  0,  0
//, 1,  0,  1,  0,  0,  0,  0,  0

///*--------------------------------- Graph 19 ----------------------------------*/

//    Eigen::MatrixXi g19_adj(11,11);
//, 0,  1,  0,  0,  0,  0
//, 1,  0,  1,  0,  0,  1
//, 0,  1,  0,  0,  0,  1
//, 0,  0,  0,  0,  1,  1
//, 0,  0,  0,  1,  0,  1
//, 0,  1,  1,  1,  1,  0

///*--------------------------------- Graph 20 ----------------------------------*/

//    Eigen::MatrixXi g20_adj(11,11);
//, 0,  1,  0,  0,  0
//, 1,  0,  1,  1,  0
//, 0,  1,  0,  1,  1
//, 0,  1,  1,  0,  1
//, 0,  0,  1,  1,  0

///*--------------------------------- Graph 21 ----------------------------------*/

//    Eigen::MatrixXi g21_adj(11,11);
//, 0,  1,  1,  1,  0,  0,  0
//, 1,  0,  1,  0,  0,  0,  0
//, 1,  1,  0,  1,  0,  1,  1
//, 1,  0,  1,  0,  1,  1,  0
//, 0,  0,  0,  1,  0,  1,  0
//, 0,  0,  1,  1,  1,  0,  0
//, 0,  0,  1,  0,  0,  0,  0

///*--------------------------------- Graph 22 ----------------------------------*/

//    Eigen::MatrixXi g22_adj(11,11);
//, 0,  1,  1,  0,  0
//, 1,  0,  1,  0,  0
//, 1,  1,  0,  1,  0
//, 0,  0,  1,  0,  1
//, 0,  0,  0,  1,  1

///*--------------------------------- Graph 23 ----------------------------------*/

//    Eigen::MatrixXi g23_adj(11,11);
//, 0,  1,  1
//, 1,  0,  1
//, 1,  1,  0

///*--------------------------------- Graph 24 ----------------------------------*/

//    Eigen::MatrixXi g24_adj(11,11);

//    g24_adj <<  0,  1,  1,  0,  1,  1,  0,  0,  0,
//                1,  0,  1,  1,  1,  0,  0,  0,  0,
//                1,  1,  0,  1,  1,  0,  1,  0,  0,
//                0,  1,  1,  0,  1,  1,  1,  1,  0,
//                1,  1,  1,  1,  0,  0,  1,  0,  0,
//                1,  0,  0,  1,  0,  0,  0,  0,  0,
//                0,  0,  1,  1,  1,  0,  0,  0,  0,
//                0,  0,  0,  1,  0,  0,  0,  0,  1,
//                0,  0,  0,  0,  0,  0,  0,  1,  0;

///*--------------------------------- Graph 25 ----------------------------------*/

//    Eigen::MatrixXi g25_adj(11,11);

//    g25_adj <<  0,  1,  1,  0,  0,
//                1,  0,  1,  1,  1,
//                1,  1,  0,  0,  0,
//                0,  1,  0,  0,  1,
//                0,  1,  0,  1,  0;

    static boost::mt19937 rng;
    rng.seed(std::time(0));

    size_t N_graphsForTraining = 9;

    size_t absoluteSuccess_Greedy  = 0;
    size_t absoluteSuccess_ICM     = 0;
    size_t absoluteSuccess_AlphaExpansion = 0;
    size_t absoluteSuccess_AlphaBetaSwap  = 0;
    size_t absoluteSuccess_LBP = 0;
    size_t absoluteSuccess_TRP = 0;
    size_t absoluteSuccess_RBP = 0;
    size_t absoluteNumberOfNodes = 0;

    size_t N_repetitions = 100;

    for ( size_t rep = 0; rep < N_repetitions; rep++ )
    {

    vector<size_t> graphs_to_train;
    size_t N_graphsAdded       = 0;

    while ( N_graphsAdded < N_graphsForTraining )
    {
        boost::uniform_int<> int_generator(0,11);
        int rand = int_generator(rng);

        if ( std::find(graphs_to_train.begin(),graphs_to_train.end(),rand)
             == graphs_to_train.end() )
        {
            graphs_to_train.push_back(rand);
            N_graphsAdded++;
        }
    }

    vector<size_t> graphs_to_test;

    for ( size_t i = 0; i < graphs.size(); i++ )
        if ( std::find(graphs_to_train.begin(),graphs_to_train.end(), i)
             == graphs_to_train.end() )
            graphs_to_test.push_back(i);

    cout << "Graphs to test: ";

    for ( size_t i = 0; i < graphs_to_test.size(); i++)
        cout << graphs_to_test[i] << " ";

    cout << endl;


//    graphs_to_train.push_back(0);
//    graphs_to_train.push_back(1);
//    graphs_to_train.push_back(2);
//    graphs_to_train.push_back(3);
//    graphs_to_train.push_back(4);
//    graphs_to_train.push_back(5);
//    graphs_to_train.push_back(6);
//    graphs_to_train.push_back(7);
//    graphs_to_train.push_back(8);
//    graphs_to_train.push_back(9);
//    graphs_to_train.push_back(10);


//    graphs_to_test.push_back(11);

    for ( size_t i = 0; i < graphs_to_train.size(); i++ )
    {
        trainingDataset.addGraph( graphs[graphs_to_train[i]] );
        trainingDataset.addGraphGroundTruth( groundTruths[graphs_to_train[i]] );
    }


/*------------------------------------------------------------------------------
 *
 *                               TRAINING!
 *
 *----------------------------------------------------------------------------*/

    UPGMpp::TTrainingOptions to;
    to.l2Regularization     = true;
    to.nodeLambda           = 10;
    to.edgeLambda           = 15;
    to.showTrainedWeights   = false;
    to.showTrainingProgress = false;

    cout << "------------------------------------------------------" << endl;
    cout << "                      TRAINING" << endl;
    cout << "------------------------------------------------------" << endl;

    trainingDataset.setTrainingOptions( to );
    int res = trainingDataset.train();

    if (res!=0) continue;


/*------------------------------------------------------------------------------
 *
 *              PREPARE A GRAPH FOR PERFORMING DECODING AND INFERENCE
 *
 *----------------------------------------------------------------------------*/

    cout << "------------------------------------------------------" << endl;
    cout << "                      TESTING" << endl;
    cout << "------------------------------------------------------" << endl;

    for ( size_t i = 0; i < graphs_to_test.size(); i++ )
    {

        graphs[graphs_to_test[i]].computePotentials();

    /*------------------------------------------------------------------------------
     *
     *                               DECODING!
     *
     *----------------------------------------------------------------------------*/

        cout << endl;
        cout << "------------------------------------------------------" << endl;
        cout << "                      DECODING" << endl;
        cout << "------------------------------------------------------" << endl;


        CDecodeICM       decodeICM;
        CDecodeICMGreedy decodeICMGreedy;        
        CDecodeAlphaExpansion decodeAlphaExpansion;
        CDecodeAlphaBetaSwap decodeAlphaBetaSawp;
        CDecodeLBP      decodeLBP;
        CDecodeTRPBP    decodeTRPBP;
        CDecodeRBP      decodeRBP;

        double totalSuccess_Greedy  = 0;
        double totalSuccess_ICM     = 0;
        double totalSuccess_AlphaExpansion = 0;
        double totalSuccess_AlphaBetaSwap  = 0;
        double totalSuccess_LBP = 0;
        double totalSuccess_TRP = 0;
        double totalSuccess_RBP = 0;

        TInferenceOptions options;
        options.maxIterations = 100;

        std::map<size_t,size_t> resultsMap;
        std::map<size_t,size_t>::iterator it;

        //std::cout << "           RESULTS ICM " << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        decodeICM.setOptions( options );
        decodeICM.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_ICM++;

        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;
        //std::cout << "         RESULTS ICM GREEDY" << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        decodeICMGreedy.setOptions( options );
        decodeICMGreedy.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_Greedy++;

        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;
        //std::cout << "      RESULTS ALPHA-EXPANSIONS" << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        options.maxIterations = 10000;
        decodeAlphaExpansion.setOptions( options );
        decodeAlphaExpansion.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_AlphaExpansion++;

        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;
        //std::cout << "        RESULTS ALPHA-BETA" << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        options.maxIterations = 10000;
        decodeAlphaBetaSawp.setOptions( options );
        decodeAlphaBetaSawp.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
        {
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_AlphaBetaSwap++;
        }

        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;
        //std::cout << "        RESULTS LBP" << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        options.maxIterations = 10000;
        decodeLBP.setOptions( options );
        decodeLBP.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_LBP++;


        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;
        //std::cout << "        RESULTS TRP" << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        options.maxIterations = 10000;
        decodeTRPBP.setOptions( options );
        decodeTRPBP.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_TRP++;

        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;
        //std::cout << "         RESULTS RBP" << std::endl;
        //std::cout << "-----------------------------------" << std::endl;

        options.maxIterations = 10000;
        decodeRBP.setOptions( options );
        //decodeRBP.decode( graphs[graphs_to_test[i]], resultsMap );

        for ( it = resultsMap.begin(); it != resultsMap.end(); it++ )
            if ( it->second == groundTruths[graphs_to_test[i]][ it->first ])
                totalSuccess_RBP++;

        //showResults( resultsMap );

        //std::cout << "-----------------------------------" << std::endl;

        double totalNumberOfNodes = graphs[graphs_to_test[i]].getNodes().size();

        //cout << "Total MaxNodePot success: " << 100*(totalSuccess_MaxNodePot / totalNumberOfNodes) << "%" << endl;
        cout << "Total Greedy     success: " << 100*(totalSuccess_Greedy / totalNumberOfNodes) << "%" << endl;
        cout << "Total ICM        success: " << 100*(totalSuccess_ICM / totalNumberOfNodes) << "%" << endl;
        //cout << "Total Exact      success: " << 100*(totalSuccess_Exact / totalNumberOfNodes) << "%" << endl;
        cout << "Total AlphaExpan success: " << 100*(totalSuccess_AlphaExpansion / totalNumberOfNodes) << "%" << endl;
        cout << "Total AlphaBetaS success: " << 100*(totalSuccess_AlphaBetaSwap / totalNumberOfNodes) << "%" << endl;
        cout << "Total LBP        success: " << 100*(totalSuccess_LBP / totalNumberOfNodes) << "%" << endl;
        cout << "Total TRP        success: " << 100*(totalSuccess_TRP / totalNumberOfNodes) << "%" << endl;
        cout << "Total RBP        success: " << 100*(totalSuccess_RBP / totalNumberOfNodes) << "%" << endl;

        absoluteNumberOfNodes           += totalNumberOfNodes;
        absoluteSuccess_Greedy          += totalSuccess_Greedy;
        absoluteSuccess_ICM             += totalSuccess_ICM;
        absoluteSuccess_AlphaExpansion  += totalSuccess_AlphaExpansion;
        absoluteSuccess_AlphaBetaSwap   += totalSuccess_AlphaBetaSwap;
        absoluteSuccess_LBP             += totalSuccess_LBP;
        absoluteSuccess_TRP             += totalSuccess_TRP;
        absoluteSuccess_RBP             += totalSuccess_RBP;

    /*------------------------------------------------------------------------------
     *
     *                               INFERENCE!
     *
     *----------------------------------------------------------------------------*/

        /*cout << endl;
        cout << "------------------------------------------------------" << endl;
        cout << "                      INFERENCE" << endl;
        cout << "------------------------------------------------------" << endl;


        std::cout << "     LOOPY BELIEF PROPAGATION " << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        CLBPInference LBPInference;

        map<size_t,VectorXd> nodeBeliefs;
        map<size_t,MatrixXd> edgeBeliefs;
        double logZ;

        LBPInference.infer( graphs[graphs_to_test[i]], nodeBeliefs, edgeBeliefs, logZ );

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

        cout << "Log Z : " << logZ << endl;*/
    }

    }

    cout << " FINAL RESULTS: " << endl;

    //cout << "Total MaxNodePot success: " << 100*(totalSuccess_MaxNodePot / totalNumberOfNodes) << "%" << endl;
    cout << "Total Greedy     success: " << 100*((double)absoluteSuccess_Greedy / absoluteNumberOfNodes) << "%" << endl;
    cout << "Total ICM        success: " << 100*((double)absoluteSuccess_ICM / absoluteNumberOfNodes) << "%" << endl;
    //cout << "Total Exact      success: " << 100*(totalSuccess_Exact / totalNumberOfNodes) << "%" << endl;
    cout << "Total AlphaExpan success: " << 100*((double)absoluteSuccess_AlphaExpansion / absoluteNumberOfNodes) << "%" << endl;
    cout << "Total AlphaBetaS success: " << 100*((double)absoluteSuccess_AlphaBetaSwap / absoluteNumberOfNodes) << "%" << endl;
    cout << "Total LBP        success: " << 100*((double)absoluteSuccess_LBP / absoluteNumberOfNodes) << "%" << endl;
    cout << "Total TRP        success: " << 100*((double)absoluteSuccess_TRP / absoluteNumberOfNodes) << "%" << endl;
    cout << "Total RBP        success: " << 100*((double)absoluteSuccess_RBP / absoluteNumberOfNodes) << "%" << endl;

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
