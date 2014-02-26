
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


#include "decoding.hpp"

using namespace UPGMpp;
using namespace std;



/*------------------------------------------------------------------------------

                                decodeICM

------------------------------------------------------------------------------*/

size_t UPGMpp::decodeICM( CGraph &graph, TOptions &options, std::map<size_t,size_t> &results )
{
    //cout << "Satarting ICM decoding..." << endl;

    // Direct access to useful vbles
    const std::vector<CNodePtr> &nodes = graph.getNodes();
    std::multimap<size_t,CEdgePtr> &edges_f = graph.getEdgesF();
    size_t N_nodes = nodes.size();

    // Initialize the results vector
    results.clear();

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials

    //cout << "Initial classes assignation" << endl;
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        size_t ID = nodes[index]->getId();

        nodes[index]->getPotentials().maxCoeff(&nodeMAP);
        //cout << "Node potentials for " << ID << ": " << nodes[index]->getPotentials().transpose() << endl;
        results[ID] = nodeMAP;
        //cout << "nodeMap" << nodeMAP << endl;
    }

    // Set the stop conditions
    bool keep_iterating = true;

    size_t iteration = 1;

    // Let's go!
    while ( (keep_iterating) && ( iteration <= options.maxIterations ) )
    {
        bool changes = false;

        // Iterate over all the nodes, and check if a more promissing class
        // is waiting for us
        for ( size_t index = 0; index < N_nodes; index++ )
        {
            Eigen::VectorXd potentials = nodes[index]->getPotentials();
            size_t ID = nodes[index]->getId();

            pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator > neighbors;

            neighbors = edges_f.equal_range(ID);

            // Iterating over the neighbors, multiplying the potential of the
            // corresponding col. in the edgePotentials, according to their class
            for ( multimap<size_t,CEdgePtr>::iterator it = neighbors.first; it != neighbors.second; it++ )
            {
                size_t neighborID;
                size_t ID1, ID2;

                CEdgePtr edgePtr( (*it).second );

                edgePtr->getNodesID(ID1,ID2);

                ( ID1 == ID ) ? neighborID = ID2 : neighborID = ID1;

                //cout << "Testing edge <" << ID << "," << neighborID << ">" << endl;
                Eigen::MatrixXd edgePotentials = edgePtr->getPotentials();

                //cout << "Edge potentials" << edgePotentials.col(results[neighborID]) << endl;

                potentials = potentials.cwiseProduct(
                            edgePotentials.col(results[neighborID])
                            );
            }

            //cout << "Potentials" << endl << potentials << endl;
            size_t class_res;
            potentials.maxCoeff(&class_res);

            if ( class_res != results[ID] )
            {
                changes = true;
                size_t previous = results[ID];
                results[ID] = class_res;

                //cout << "Changing node " << index << " from " << previous << " to " << class_res << endl;
            }

        }

        // If any change done, stop iterating, convergence achieved!
        if ( !changes )
            keep_iterating = false;

        iteration++;
    }

    if ( options.maxIterations < iteration )
        return 0;
    else
        return 1;

    // TODO: It could be interesting return the case of stopping iterating
}


/*------------------------------------------------------------------------------

                              decodeGreedy

------------------------------------------------------------------------------*/

size_t UPGMpp::decodeICMGreedy( CGraph &graph,
                                TOptions &options,
                                std::map<size_t,size_t> &results )
{

    // Direct access to useful vbles
    const std::vector<CNodePtr> &nodes = graph.getNodes();
    std::multimap<size_t,CEdgePtr> &edges_f = graph.getEdgesF();
    size_t N_nodes = nodes.size();


    // Intilize the results vector
    results.clear();

    Eigen::VectorXd v_potentials;
    v_potentials.resize( N_nodes );

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        size_t ID = nodes[index]->getId();

        nodes[index]->getPotentials().maxCoeff(&nodeMAP);
        results[ID] = nodeMAP;
        v_potentials(index) = 0; //std::numeric_limits<double>::min();
    }

    // Set the stop conditions
    bool keep_iterating = true;

    size_t iteration = 1;
    Eigen::VectorXd v_new_potentials;
    v_new_potentials.resize( N_nodes );
    map<size_t,size_t> new_results;

    // Let's go!
    while ( (keep_iterating) && ( iteration <= options.maxIterations) )
    {

        // Iterate over all the nodes, and check if a more promissing class
        // is waiting for us
        for ( size_t index = 0; index < N_nodes; index++ )
        {
            size_t ID = nodes[index]->getId();

            Eigen::VectorXd potentials = nodes[index]->getPotentials();

            pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator > neighbors;

            neighbors = edges_f.equal_range(ID);

            // Iterating over the neighbors, multiplying the potential of the
            // corresponding col. in the edgePotentials, according to their class
            for ( multimap<size_t,CEdgePtr>::iterator it = neighbors.first; it != neighbors.second; it++ )
            {
                size_t neighborID;
                size_t ID1, ID2;

                CEdgePtr edgePtr( (*it).second );

                edgePtr->getNodesID(ID1,ID2);

                ( ID1 == ID ) ? neighborID = ID2 : neighborID = ID1;


                Eigen::MatrixXd edgePotentials = edgePtr->getPotentials();

                potentials = potentials.cwiseProduct( edgePotentials.col(results[neighborID]) );
            }

            size_t class_res;
            double max_potential = potentials.maxCoeff(&class_res);

            v_new_potentials(index) = max_potential;
            new_results[ID] = class_res;

        }

      //  cout << "Iteration Potentials" << endl << v_new_potentials << endl;

        Eigen::VectorXd difference = v_new_potentials - v_potentials;

       // cout << "Difference" << difference << endl;
        size_t node;
        double max_difference = difference.maxCoeff(&node);

        if ( max_difference > 0 )
        {
            v_potentials(node) = v_new_potentials(node);
            size_t IDNodeToChange = nodes[node]->getId();
            results[IDNodeToChange] = new_results[IDNodeToChange];
        }
        else
            keep_iterating = false;
    }

    if ( options.maxIterations < iteration )
        return 0;
    else
        return 1;


    // TODO: It could be interesting return the case of stopping iterating
}


/*------------------------------------------------------------------------------

                                decodeExact

------------------------------------------------------------------------------*/

void decodeExactRec( CGraph &graph,
                const std::map<size_t,vector<size_t> > &mask,
                size_t index,
                map<size_t,size_t> &nodesAndValuesToTest,
                std::map<size_t,size_t> &results,
                double &maxLikelihood);

size_t UPGMpp::decodeExact(CGraph &graph, TOptions &options, std::map<size_t,size_t> &results, const std::map<size_t, vector<size_t> > &mask )
{

    results.clear();

    double maxLikelihood = std::numeric_limits<double>::min();
    map<size_t,size_t>  nodesAndValuesToTest;
    std::vector<CNodePtr> &nodes = graph.getNodes();
    size_t index = 0;

    decodeExactRec( graph, mask, index, nodesAndValuesToTest, results, maxLikelihood);

    return 1;
}

void decodeExactRec( CGraph &graph,
                const std::map<size_t,vector<size_t> > &mask,
                size_t index,
                map<size_t,size_t> &nodesAndValuesToTest,
                std::map<size_t,size_t> &results,
                double &maxLikelihood)
{
    const CNodePtr node = graph.getNode( index );
    const size_t  nodeID = node->getId();
    const CNodeTypePtr &nodeType = node->getType();
    size_t nodeClasses = nodeType->getClasses();
    size_t totalNodes = graph.getNodes().size();
    vector<size_t> classesToCheck;

    for ( size_t i = 0; i < nodeClasses; i++ )
        classesToCheck.push_back(i);

    if ( !mask.empty() )
    {
        if ( mask.count( nodeID ) )
            classesToCheck = mask.at(nodeID);
    }

    for ( size_t i = 0; i < classesToCheck.size(); i++ )
    {
        size_t classToCheck = classesToCheck[i];

        if ( index == totalNodes - 1 )
        {
            nodesAndValuesToTest[nodeID] = classToCheck;
            double likelihood = graph.getUnnormalizedLogLikelihood( nodesAndValuesToTest );

            cout << "Testing..." ;
            for ( map<size_t,size_t>::iterator it = nodesAndValuesToTest.begin(); it != nodesAndValuesToTest.end(); it++ )
            {
                std::cout << "[ID:" << it->first << " ,value " << it->second << "]";
            }
            cout << " Likelihood: " << likelihood << " Max: " << maxLikelihood<< endl ;

            if ( likelihood > maxLikelihood )
            {
                results = nodesAndValuesToTest;
                maxLikelihood = likelihood;
            }
        }
        else
        {
            nodesAndValuesToTest[nodeID] = classToCheck;
            decodeExactRec( graph, mask, index + 1, nodesAndValuesToTest, results, maxLikelihood);
        }
    }
}
