
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
 * In this example it is shown how to use the library when you have already the
 * weights to use to compute the potentials, maybe obtained from your favourite
 * optimizer external to this library.
 *
 * The example steps are:
 * 1. Loading of the pre-computed weights.
 * 2. Set of the node and edge types using that weights.
 * 3. Building of an example graph.
 * 4. MAP inference computation for that graph.
 *
 *---------------------------------------------------------------------------*/

int main (int argc, char* argv[])
{

    cout << endl;
    cout << "       " << "PRE-COMPUTED WEIGHTS EXAMPLE";
    cout << endl << endl;


/*------------------------------------------------------------------------------
 *
 *                              PREPARE DATA
 *
 *----------------------------------------------------------------------------*/

    // Load the vector of weights. Please, don't you get scared by the size of
    // the vector ;)

    Eigen::Matrix<double,372,1> w;

    w <<    -0.015796075927673,
             0.017436684385441,
            -0.025762438163845,
             0.013590785778994,
             0.010510736568731,
            -0.033229384613447,
             0.006766540824796,
            -0.002323198727633,
             0.004398261360731,
             0.001281664427650,
             0.025319369161658,
            -0.002192945075403,
            -0.055802919966258,
            -0.000746930558618,
             0.024830027559054,
            -0.029193307801860,
             0.002970972903736,
            -0.006145922316205,
             0.035578345210637,
            -0.005269868044002,
            -0.000591389714957,
             0.016580340741208,
             0.021713814217721,
            -0.003923162230456,
             0.030525841452530,
             0.036315711831236,
             0.026459871506641,
            -0.024954733392817,
            -0.037424386716078,
            -0.028492816438586,
             0.007596289698958,
             0.001491563967207,
            -0.040732548452933,
            -0.002690417421435,
             0.033140605507755,
            -0.001234981542476,
             0.018488325369830,
             0.020497765441363,
             0.001546248227346,
             0.004902986668663,
             0.002202098948971,
            -0.006042901822504,
            -0.040684476739179,
            -0.006652142121025,
             0.008191557831540,
             0.018584316150295,
            -0.016151219591161,
            -0.004882558364139,
             0.014239478290819,
            -0.038631660411504,
            -0.006515173077460,
             0.033068157540358,
             0.031735591664346,
             0.072825909188184,
             0.026878334310987,
            -0.015855117458242,
            -0.000188532253995,
            -0.046935533104831,
            -0.039470964652775,
            -0.031150490035886,
            -0.005265011023655,
             0.012898321088497,
            -0.007848734543238,
             0.006839544338609,
             0.008644431541238,
            -0.018145893726497,
            -0.006525518224589,
            -0.001002037615392,
            -0.006384040747812,
            -0.007763158392390,
             0.000358832933551,
            -0.000483566476876,
            -0.019640518707743,
             0.011145388929904,
            -0.014658011675315,
            -0.015960440286587,
             0.002089642204607,
            -0.014025130990468,
             0.000821434658648,
            -0.004556461021615,
            -0.005662224526657,
             0.012771135956806,
            -0.000411996449483,
            -0.012335004063795,
             0.020259804149740,
             0.025967144868745,
            -0.025664546965946,
             0.000828051167134,
            -0.001398179342398,
             0.009115516066249,
             0.006255533643945,
            -0.002964686782492,
            -0.001388690843755,
            -0.002454136813323,
            -0.007293423307926,
             0.025572507695483,
            -0.006827576318489,
             0.000765559270736,
            -0.001406214231548,
            -0.001934946393004,
             0.007917967754125,
            -0.000099584368759,
            -0.002271930995637,
             0.027727400741483,
            -0.004767219065907,
            -0.000964586869460,
            -0.002063646912666,
            -0.002689857157213,
             0.002795717651851,
            -0.000344750542966,
            -0.007705197654944,
             0.001518706461924,
            -0.001476259111645,
             0.011368151114769,
             0.004385651642566,
             0.002157932022957,
            -0.000747501522601,
            -0.001491983472985,
             0.001899090315518,
            -0.001529342487421,
            -0.001900121074064,
             0.007849919666639,
            -0.000182069758668,
            -0.000001483599280,
             0.000100655676808,
            -0.000241969865454,
             0.003067257466624,
            -0.000000554095940,
            -0.000112252016776,
            -0.000285116745532,
             0.004665260444927,
            -0.000022863272442,
            -0.000190900253667,
             0.002342522921690,
            -0.000027918955712,
            -0.000981524264943,
            -0.000030298860100,
                             0,
            -0.008308226315263,
             0.017141507393390,
            -0.016669436003780,
             0.004923929847695,
             0.017099904124850,
             0.027806214243581,
            -0.007462367919162,
            -0.001743820959203,
            -0.007230557747141,
            -0.011822639098908,
            -0.004573357674455,
            -0.000846695589386,
             0.000037947814771,
             0.015228115587298,
             0.012742641487841,
             0.007760903541112,
             0.012210758967415,
             0.008554560494970,
             0.001063306113283,
             0.007413543772313,
             0.007222592636890,
             0.012202112461687,
            -0.000896304218886,
            -0.015039701664155,
             0.000977690899165,
            -0.051110776771945,
             0.017067464794951,
             0.001615854425569,
             0.001554305573399,
            -0.017390445827923,
             0.004534391288675,
             0.010070589273274,
            -0.000770318768127,
            -0.014116050163109,
             0.009335519956994,
            -0.032121425446905,
             0.015479996925199,
             0.000563979254526,
             0.010790654991295,
             0.009861528023874,
             0.012215565629001,
            -0.000205764832770,
            -0.039622545632236,
            -0.008944363666817,
            -0.004078054153285,
            -0.001718596106110,
            -0.000658094320795,
            -0.002221182951329,
             0.003083956787875,
            -0.000477079570223,
            -0.012178299431262,
             0.007592970880702,
            -0.000955688300453,
             0.005243496164299,
             0.008841997368393,
             0.005022260776456,
            -0.000495774007465,
            -0.022688811504057,
             0.005988701906595,
             0.004144165348417,
            -0.006629059628556,
             0.004103394170627,
            -0.000532516547613,
            -0.000003050262517,
             0.003516004583309,
            -0.001059840264079,
             0.000086387595131,
            -0.000001503762250,
            -0.003953328991817,
            -0.000921128000080,
             0.005403129917255,
            -0.000071840288359,
            -0.003414060489917,
             0.003729522578142,
            -0.000109396661735,
            -0.001148682207560,
            -0.000040781850591,
                             0,
                             0,
                             0,
            -0.000405496115041,
                             0,
                             0,
                             0,
            -0.003171161241103,
                             0,
            -0.000184271664046,
            -0.000297064093442,
                             0,
                             0,
                             0,
            -0.028274507298781,
                             0,
                             0,
                             0,
            -0.000962284264013,
                             0,
            -0.000128318851364,
            -0.000138890101225,
                             0,
                             0,
            -0.005775692988707,
            -0.014524789835743,
            -0.032978898158974,
            -0.004032224952414,
             0.045707329263042,
            -0.001027345982652,
             0.039368757751744,
             0.033139527576618,
            -0.014492944917985,
            -0.000564683466302,
                             0,
                             0,
                             0,
            -0.000041868901518,
                             0,
            -0.000002142580228,
            -0.000006397173521,
                             0,
                             0,
                             0,
                             0,
            -0.000101646624145,
                             0,
            -0.000005142334432,
            -0.000013651741369,
                             0,
                             0,
                             0,
            -0.005313712267491,
                             0,
            -0.000484098807673,
            -0.000703571651653,
                             0,
                             0,
            -0.000367243623386,
            -0.000096108589530,
            -0.000113947449102,
            -0.000473762444521,
            -0.002828335403568,
            -0.000034991818873,
                             0,
            -0.000004890157346,
            -0.000009157067588,
                             0,
                             0,
            -0.000003344804987,
            -0.000033073355580,
            -0.000216771062841,
            -0.000001939175878,
            -0.000047067970957,
            -0.000350121673469,
            -0.000004053979958,
                             0,
                             0,
                             0,
             0.000976688748611,
            -0.000651650374690,
            -0.009197430145412,
            -0.000232642793496,
            -0.000770774849181,
             0.008011672964031,
            -0.003445295472267,
            -0.000545238716393,
            -0.002602242463159,
            -0.002875723281208,
            -0.002117080481829,
            -0.000252875980957,
            -0.027101837766538,
            -0.001125710563218,
             0.016441910238017,
             0.022684877265108,
            -0.008712686179205,
            -0.000948816295119,
             0.000045046448254,
             0.005686952827387,
             0.008441362830451,
             0.000511582769385,
            -0.000461293588957,
            -0.008569496593025,
             0.002769087238114,
             0.005543494954474,
             0.017037855374363,
            -0.002241815945984,
            -0.000150433220937,
             0.001754912242996,
             0.000674356867003,
            -0.003495004366438,
            -0.000545910956332,
            -0.003310192403966,
             0.025231761401792,
             0.007162484445454,
             0.002963502912256,
             0.000535437043446,
             0.002085601055659,
             0.001339995527218,
             0.001764644529452,
            -0.000126084537324,
            -0.041956153921693,
             0.005429899358325,
             0.000986204153323,
            -0.000843381519577,
             0.005776330022315,
            -0.001995676241537,
            -0.005236772325952,
            -0.000315286932582,
            -0.007240730512971,
            -0.003420302072310,
            -0.000545810780823,
             0.002553470173170,
            -0.000108132939728,
            -0.003554662234403,
            -0.000269372799974,
            -0.052388876579859,
             0.000935968967473,
             0.031028176857936,
             0.032384460381622,
            -0.000975417185917,
            -0.000256373816803,
            -0.000000866667153,
             0.000554804147490,
            -0.000286316009088,
             0.000730642060639,
            -0.000000375447410,
            -0.006850032629308,
             0.002514189803194,
             0.000042604733048,
            -0.000050642839180,
            -0.008406174208452,
             0.000056878612132,
            -0.000063525736370,
            -0.000399803975283,
            -0.000011932602130,
                             0;


/*------------------------------------------------------------------------------
 *
 *                      SET OF NODE AND EDGE TYPES
 *
 *----------------------------------------------------------------------------*/

    //
    // Node types
    //


    //
    // Feature multipliers
    //

    // In feature engineering, usually features are multiplied by a factor in
    // order to scale them and do classes more distinguishable. The previous
    // weights were learnt using these factors, so we have to employ them again
    // while recognizing a new scene in order to make it consistent.

    Eigen::VectorXd nodeFeatMultiplierFactor(5);
    nodeFeatMultiplierFactor << 50, 90, 60, 50, 50;

    // This matrix maps the node weights into the vector of weights w.
    Eigen::MatrixXi nodeMapAux;

    nodeMapAux.resize(5,12); // 12 classes, 5 features per class

    nodeMapAux << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60;

    Eigen::MatrixXi nodeMap = nodeMapAux.transpose();
    Eigen::MatrixXd nodeWeights;

    size_t N_rows = nodeMap.rows();
    size_t N_cols = nodeMap.cols();

    nodeWeights.resize(N_rows,N_cols);

    // Build the nodeWeights matrix

    for ( size_t i = 0; i < N_rows; i++ )
        for ( size_t j = 0; j < N_cols; j++ )
        {
            nodeWeights(i,j) = w(nodeMap(i,j)-1);
        }


    // Ok, create the node type!

    size_t N_classes = nodeWeights.rows();
    size_t N_nodeFeatures = nodeWeights.cols();

    CNodeTypePtr nodeType1 ( new CNodeType( N_classes,
                                            N_nodeFeatures,
                                            string("object")  ) );

    std::string label("Object");
    nodeType1->setLabel( label );

    nodeType1->setWeights( nodeWeights );

    //
    // Types of edges
    //

    std::vector<Eigen::MatrixXi> edgeWeightsMap;

    // Edge features to use

    Eigen::VectorXd edgeFeatMultiplierFactor(4);
    edgeFeatMultiplierFactor << 1, 1, 50, 1;

    // Here we can choose the edge features that we want to use. 1 means use
    // this feature, 0 please leave it away.
    Eigen::VectorXi edgeFeaturesToUse(6);
    edgeFeaturesToUse << 1, 1, 0, 0, 1, 1;

    size_t N_edgeFeatures = edgeFeaturesToUse.sum();

    edgeWeightsMap.resize(N_edgeFeatures);

    for (size_t i = 0; i < N_edgeFeatures; i++ )
        edgeWeightsMap[i].resize(N_classes,N_classes);

    size_t f = 61;
    size_t first_f = 61;

    for ( size_t edge_feat = 0; edge_feat < N_edgeFeatures; edge_feat++ )
    {

        for ( size_t c1 = 0; c1 < N_classes; c1++ ) // rows
        {
            f++;

            for ( size_t c2 = c1+1; c2 < N_classes; c2++ )  // cols
            {
                     edgeWeightsMap[edge_feat](c1,c2) = f;
                     edgeWeightsMap[edge_feat](c2,c1) = f;
                     f++;
                     //std::cout << "f: " << f << std::endl;
            }
        }

            size_t previousW = first_f;

            edgeWeightsMap[edge_feat](0,0) = first_f;

            for ( size_t c3 = 1; c3 < N_classes; c3++ )
            {
                f = previousW + (N_classes - c3) + 1;

                //std::cout << "F: " << f << std::endl;
                edgeWeightsMap[edge_feat](c3,c3) = f;
                previousW = f;
            }

            first_f = edgeWeightsMap[edge_feat](N_classes-1,N_classes-1)+1;
            f++;

    }

    std::vector<Eigen::MatrixXd> edgeWeights;
    edgeWeights.resize(N_edgeFeatures);

    for (size_t i = 0; i < N_edgeFeatures; i++ )
        edgeWeights[i].resize(N_classes,N_classes);

    // Replace indexes by their values in w

    for ( size_t edge_feat = 0; edge_feat < N_edgeFeatures; edge_feat++ )
        for ( size_t c1 = 0; c1 < N_classes; c1++ ) // rows
            for ( size_t c2 = 0; c2 < N_classes; c2++ )  // cols
            {
                edgeWeights[edge_feat](c1,c2) = w(edgeWeightsMap[edge_feat](c1,c2)-1);
            }


    CEdgeTypePtr edgeType1 ( new CEdgeType( N_edgeFeatures,
                                            nodeType1,
                                            nodeType1,
                                            string("Object-object")) );
    edgeType1->setWeights( edgeWeights );


/*------------------------------------------------------------------------------
 *
 *                      BUILDING OF AN EXAMPLE GRAPH
 *
 *----------------------------------------------------------------------------*/

    CGraph myGraph;

    //
    // Nodes
    //

    Eigen::Matrix<double,11,5> node_features; // 11 objects, 5 features per object

    // Features are: orientation centroid_z area elongation bias

    node_features << 1, 0.725256, 0.347833, 2.32114, 1,
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

    size_t N_nodes = node_features.rows();

    // Insert the nodes into the graph

    for ( size_t i = 0; i < N_nodes; i++ )
    {
        Eigen::VectorXd features = node_features.row(i);

        Eigen::VectorXd scaledFeatures =
                features = features.cwiseProduct( nodeFeatMultiplierFactor );

        CNodePtr nodePtr ( new CNode( nodeType1, // type
                                      scaledFeatures )// node features
                                    );

        myGraph.addNode( nodePtr );
    }


    //
    // Edges
    //

    Eigen::Matrix<float,28,2> edges; // A total of 28 objects between the 11 objects

    edges << 1, 2,
    1, 3,
    1, 4,
    1, 5,
    1, 6,
    1, 9,
    1, 10,
    2, 3,
    2, 9,
    2, 10,
    4, 5,
    4, 6,
    4, 7,
    4, 8,
    4, 9,
    4, 11,
    5, 6,
    5, 7,
    5, 8,
    5, 9,
    5, 10,
    5, 11,
    6, 7,
    7, 8,
    7, 9,
    7, 11,
    8, 11,
    9, 10;

    Eigen::Matrix<int,28,1> pre_computed_edge_features;

    pre_computed_edge_features << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // Create edges

    size_t N_edges = 28;

    for ( size_t i = 0; i < N_edges; i++ )
    {
        Eigen::Vector2f nodes = edges.row(i);
        Eigen::Vector2f ones(1,1);
        nodes = nodes - ones;

        CNodePtr n1( myGraph.getNode(nodes(0)) );
        CNodePtr n2( myGraph.getNode(nodes(1)) );

        Eigen::Matrix<double,Eigen::Dynamic,1> edgeFeatures;
        edgeFeatures.resize(N_edgeFeatures);

        Eigen::VectorXd feat1 = n1->getFeatures();
        Eigen::VectorXd feat2 = n2->getFeatures();

        // Computation of the edge features;

        size_t feature = 0;


        // Perpendicularity
        if ( edgeFeaturesToUse(0) )
        {
            edgeFeatures(feature) = std::abs(feat1(0) - feat2(0));
            feature++;
        }

        // Height distance between centers
        if ( edgeFeaturesToUse(1) )
        {
            edgeFeatures(feature) = std::abs((float)feat1(1) - (float)feat2(1));
            feature++;
        }

        // Ratio between areas
        if ( edgeFeaturesToUse(2) )
        {
            float a1 = feat1(2);
            float a2 = feat2(2);

            if ( a1 < a2 )
            {
                float aux = a2;
                a2 = a1;
                a1 = aux;
            }

            edgeFeatures(feature) = a1 / a2;
            feature++;
        }

        //  Difference between elongations
        if  ( edgeFeaturesToUse(3) )
        {
            edgeFeatures(feature) = std::abs(feat1(3) - feat2(3));
            feature++;
        }

        // Use the isOn semantic relation?

        if ( edgeFeaturesToUse(4) )
        {
            edgeFeatures(feature) = pre_computed_edge_features(i);
            feature++;
        }

        // Use bias feature in edges?
        if ( edgeFeaturesToUse(5) )
        {
            edgeFeatures(feature) = 1;
            feature++;
        }


        Eigen::VectorXd scaledFeatures =
                 edgeFeatures.cwiseProduct( edgeFeatMultiplierFactor );

        CEdgePtr edgePtr ( new CEdge( n1, n2, edgeType1, scaledFeatures) );

        myGraph.addEdge( edgePtr );
    }


/*------------------------------------------------------------------------------
 *
 *                             MAP INFERENCE
 *
 *----------------------------------------------------------------------------*/

    CICMInferenceMAP            decodeICM;
    CRestartsInferenceMAP       decodeWithRestarts;
    CAlphaExpansionInferenceMAP decodeAlpha;
    CAlphaBetaSwapInferenceMAP  decodeAlphaBeta;
    CMaxNodePotInferenceMAP     decodeMax;
    CTRPBPInferenceMAP          decodeTRPBP;
    CRBPInferenceMAP            decodeRBP;

    TInferenceOptions    options;
    options.maxIterations = 100;
    options.convergency   = 0.0001;

    myGraph.computePotentials();

    std::map<size_t,size_t> results;

    decodeICM.setOptions( options );
    decodeICM.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "                 RESULTS ICM " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    std::map<size_t,size_t>::iterator it;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }

    options.particularS["method"] = "ICM";
    options.particularD["numberOfRestarts"] = 1000;
    decodeWithRestarts.setOptions( options );
    decodeWithRestarts.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "            RESULTS ICM with restarts " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }

    decodeAlpha.setOptions( options );
    decodeAlpha.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "            RESULTS ALPHA EXPANSION " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }

    options.maxIterations = 100;
    decodeAlphaBeta.setOptions( options );
    decodeAlphaBeta.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "           RESULTS ALPHA BETA SWAP " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }

    decodeMax.setOptions( options );
    decodeMax.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "           RESULTS MAX NODE POTENTIAL " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }

    decodeTRPBP.setOptions( options );
    decodeTRPBP.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "               RESULTS TRPBP " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }

    decodeRBP.setOptions( options );
    decodeRBP.infer( myGraph, results);

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "                RESULTS RBP " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    for ( it = results.begin(); it != results.end(); it++ )
    {
        std::cout << "[" << it->first << "] " << it->second << std::endl;
    }
    // We are ready to go for a walk :)

    return 0;
}

