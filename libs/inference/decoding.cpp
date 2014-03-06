
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

#include <stdio.h>
#include "decoding.hpp"

using namespace UPGMpp;
using namespace std;
using namespace Eigen;


/*------------------------------------------------------------------------------

                               CDecodeICM

------------------------------------------------------------------------------*/

void CDecodeICM::decode( CGraph &graph, std::map<size_t,size_t> &results )
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
        size_t ID = nodes[index]->getID();

        nodes[index]->getPotentials().maxCoeff(&nodeMAP);
        //cout << "Node potentials for " << ID << ": " << nodes[index]->getPotentials().transpose() << endl;
        results[ID] = nodeMAP;
        //cout << "nodeMap" << nodeMAP << endl;
    }

    // Set the stop conditions
    bool keep_iterating = true;

    size_t iteration = 1;

    // Let's go!
    while ( (keep_iterating) && ( iteration <= m_options.maxIterations ) )
    {
        bool changes = false;

        // Iterate over all the nodes, and check if a more promissing class
        // is waiting for us
        for ( size_t index = 0; index < N_nodes; index++ )
        {
            Eigen::VectorXd potentials = nodes[index]->getPotentials();
            size_t nodeID = nodes[index]->getID();

            pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator > neighbors;

            neighbors = edges_f.equal_range(nodeID);

            // Iterating over the neighbors, multiplying the potential of the
            // corresponding col. in the edgePotentials, according to their class
            for ( multimap<size_t,CEdgePtr>::iterator it = neighbors.first; it != neighbors.second; it++ )
            {
                size_t neighborID;
                size_t ID1, ID2;

                CEdgePtr edgePtr( (*it).second );

                edgePtr->getNodesID(ID1,ID2);
                Eigen::MatrixXd edgePotentials = edgePtr->getPotentials();

                if ( ID1 == nodeID )
                {
                    neighborID = ID2;
                    potentials = potentials.cwiseProduct(
                                 edgePotentials.col(results[neighborID])
                                 );
                }
                else
                {
                    neighborID = ID1;
                    potentials = potentials.cwiseProduct(
                                 edgePotentials.row(results[neighborID]).transpose()
                                 );
                }

                //cout << "Testing edge <" << ID << "," << neighborID << ">" << endl;


                //cout << "Edge potentials" << edgePotentials.col(results[neighborID]) << endl;


            }

            //cout << "Potentials" << endl << potentials << endl;
            size_t class_res;
            potentials.maxCoeff(&class_res);

            if ( class_res != results[nodeID] )
            {
                changes = true;
                size_t previous = results[nodeID];
                results[nodeID] = class_res;

                //cout << "Changing node " << index << " from " << previous << " to " << class_res << endl;
            }

        }

        // If any change done, stop iterating, convergence achieved!
        if ( !changes )
            keep_iterating = false;

        iteration++;
    }

    // TODO: It could be interesting return the case of stopping iterating
}


/*------------------------------------------------------------------------------

                              CDecodeGreedy

------------------------------------------------------------------------------*/

void CDecodeICMGreedy::decode( CGraph &graph,
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
        size_t ID = nodes[index]->getID();

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
    while ( (keep_iterating) && ( iteration <= m_options.maxIterations) )
    {

        // Iterate over all the nodes, and check if a more promissing class
        // is waiting for us
        for ( size_t index = 0; index < N_nodes; index++ )
        {
            size_t ID = nodes[index]->getID();

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
                Eigen::MatrixXd edgePotentials = edgePtr->getPotentials();

                if ( ID1 == ID )
                {
                    neighborID = ID2;
                    potentials = potentials.cwiseProduct(
                                 edgePotentials.col(results[neighborID])
                                 );
                }
                else
                {
                    neighborID = ID1;
                    potentials = potentials.cwiseProduct(
                                 edgePotentials.row(results[neighborID]).transpose()
                                 );
                }
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
            size_t IDNodeToChange = nodes[node]->getID();
            results[IDNodeToChange] = new_results[IDNodeToChange];
        }
        else
            keep_iterating = false;
    }

    // TODO: It could be interesting return the case of stopping iterating
}


/*------------------------------------------------------------------------------

                               CDecodeExact

------------------------------------------------------------------------------*/

void decodeExactRec( CGraph &graph,
                const std::map<size_t,vector<size_t> > &mask,
                size_t index,
                map<size_t,size_t> &nodesAndValuesToTest,
                std::map<size_t,size_t> &results,
                double &maxLikelihood);

void CDecodeExact::decode(CGraph &graph, std::map<size_t,size_t> &results )
{

    results.clear();

    double maxLikelihood = std::numeric_limits<double>::min();
    map<size_t,size_t>  nodesAndValuesToTest;
    std::vector<CNodePtr> &nodes = graph.getNodes();
    size_t index = 0;

    decodeExactRec( graph, m_mask, index, nodesAndValuesToTest, results, maxLikelihood);

}

void decodeExactRec( CGraph &graph,
                const std::map<size_t,vector<size_t> > &mask,
                size_t index,
                map<size_t,size_t> &nodesAndValuesToTest,
                std::map<size_t,size_t> &results,
                double &maxLikelihood)
{
    const CNodePtr node = graph.getNode( index );
    const size_t  nodeID = node->getID();
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

            /*cout << "Testing..." ;
            for ( map<size_t,size_t>::iterator it = nodesAndValuesToTest.begin(); it != nodesAndValuesToTest.end(); it++ )
            {
                std::cout << "[ID:" << it->first << " ,value " << it->second << "]";
            }
            cout << " Likelihood: " << likelihood << " Max: " << maxLikelihood<< endl ;*/

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


/*------------------------------------------------------------------------------

                                CDecodeLBP

------------------------------------------------------------------------------*/

void CDecodeLBP::decode( CGraph &graph,
                         std::map<size_t,size_t> &results )
{
    results.clear();
    const vector<CNodePtr> nodes = graph.getNodes();
    const vector<CEdgePtr> edges = graph.getEdges();
    multimap<size_t,CEdgePtr> edges_f = graph.getEdgesF();

    size_t N_nodes = nodes.size();
    size_t N_edges = edges.size();

    vector<vector<VectorXd> > messages;
    messagesLBP( graph, m_options, messages );

    //cout << "Convergency achieved in " << iteration << " interations";
    //cout << " of a maximum of " << m_options.maxIterations << endl;

    //
    // Now that we have the messages, compute the final beliefs and fill the
    // results map.
    //

    for ( size_t nodeIndex = 0; nodeIndex < N_nodes; nodeIndex++ )
    {
        const CNodePtr nodePtr = graph.getNode( nodeIndex );
        size_t nodeID          = nodePtr->getID();
        VectorXd nodePotPlusIncMsg = nodePtr->getPotentials();

        pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator > neighbors;

        neighbors = edges_f.equal_range(nodeID);

        //
        // Get the messages for all the neighbors, and multiply them with the node potential
        //
        for ( multimap<size_t,CEdgePtr>::iterator itNeigbhor = neighbors.first;
              itNeigbhor != neighbors.second;
              itNeigbhor++ )
        {
            CEdgePtr edgePtr( (*itNeigbhor).second );
            size_t edgeIndex = graph.getEdgeIndex( edgePtr->getID() );

            if ( !edgePtr->getNodePosition( nodeID ) ) // nodeID is the first node in the edge
                nodePotPlusIncMsg = nodePotPlusIncMsg.cwiseProduct(messages[ edgeIndex ][ 1 ]);
            else // nodeID is the second node in the dege
                nodePotPlusIncMsg = nodePotPlusIncMsg.cwiseProduct(messages[ edgeIndex ][ 0 ]);
        }

        // Normalize
        nodePotPlusIncMsg = nodePotPlusIncMsg / nodePotPlusIncMsg.sum();

        // Now the class with the higher value is the boss!
        size_t nodeMAP;

        nodePotPlusIncMsg.maxCoeff(&nodeMAP);

        results[nodeID] = nodeMAP;
    }

}
