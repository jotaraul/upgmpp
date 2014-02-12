
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


#include "dataTypes.h"

using namespace UPGMplusplus;
using namespace std;


/*------------------------------------------------------------------------------

                              computePotentials

------------------------------------------------------------------------------*/

void CGraph::computePotentials()
{
    // Method steps:
    //  1. Compute node potentials
    //  2. Compute edge potentials

    //
    //  1. Node potentials
    //

    std::vector<CNode>::iterator it;

    //cout << "NODE POTENTIALS" << endl;

    for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
    {
        // Get the node type
        size_t type = it->getType().getID();

        // Compute the node potentials according to the node type and its
        // extracted features
        Eigen::VectorXd potentials = it->getType().getWeights() * it->getFeatures();

        potentials = potentials.array().exp();

        //std::cout << potentials << std::endl << "----------------" << std::endl;

        it->setPotentials( potentials );

    }

    //
    //  2. Edge potentials
    //

    std::vector<CEdge>::iterator it2;

    //cout << "EDGE POTENTIALS" << endl;

    for ( it2 = m_edges.begin(); it2 != m_edges.end(); it2++ )
    {
        // Get the edge type, its extracted features and the number of them
        size_t type = it2->getType().getID();

        Eigen::VectorXd v_feat = it2->getFeatures();
        size_t num_feat = it2->getType().getWeights().size();

        // Compute the potential for each feature, and sum up them to obtain
        // the desired edge potential
        std::vector<Eigen::MatrixXd>    potentials_per_feat(num_feat);
        Eigen::MatrixXd potentials;


        for ( size_t feat = 0; feat < num_feat; feat++ )
        {
            potentials_per_feat.at(feat) = it2->getType().getWeights()[feat]*v_feat(feat);

            if ( !feat )
                potentials = potentials_per_feat[feat];
            else
                potentials += potentials_per_feat[feat];
        }

        potentials = potentials.array().exp();


        //std::cout <<  potentials << std::endl;
        //cout << "=======================================" << endl;

        it2->setPotentials ( potentials );
    }

}


/*------------------------------------------------------------------------------

                                decodeICM

------------------------------------------------------------------------------*/

void CGraph::decodeICM( std::vector<size_t> &results )
{
    cout << "Satarting ICM decoding..." << endl;
    // Intilize the results vector
    size_t N_nodes = m_nodes.size();
    results.clear();
    results.resize(N_nodes);

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials

    cout << "Initial classes assignation" << endl;
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        m_nodes[index].getPotentials().maxCoeff(&nodeMAP);
        results[index] = nodeMAP;
        cout << nodeMAP << endl;
    }

    // Set the stop conditions
    bool keep_iterating = true;

    m_TOptions.ICM_maxIterations = 100;
    size_t iteration = 0;

    // Let's go!
    while ( (keep_iterating) && ( iteration < m_TOptions.ICM_maxIterations) )
    {
        bool changes = false;

        // Iterate over all the nodes, and check if a more promissing class
        // is waiting for us
        for ( size_t index = 0; index < N_nodes; index++ )
        {
            Eigen::VectorXd potentials = m_nodes[index].getPotentials();
            size_t N_neighbors         = m_edges_f[index].size();

            // Iterating over the neighbors, multiplying the potential of the
            // corresponding col. in the edgePotentials, according to their class
            for ( size_t neighbor = 0; neighbor < N_neighbors; neighbor++ )
            {
                size_t edgeID     = m_edges_f[index][neighbor];

                size_t neighborID;
                size_t ID1, ID2;

                m_edges[edgeID].getNodesID(ID1,ID2);

                ( ID1 == index ) ? neighborID = ID2 : neighborID = ID1;

                cout << "Testing edge <" << index << "," << neighborID << ">" << endl;
                Eigen::MatrixXd edgePotentials = m_edges[edgeID].getPotentials();

                potentials = potentials.cwiseProduct(
                            edgePotentials.col(results[neighborID])
                            );
            }

            //cout << "Potentials" << endl << potentials << endl;
            size_t class_res;
            potentials.maxCoeff(&class_res);

            if ( class_res != results[index] )
            {
                changes = true;
                size_t previous = results[index];
                results[index] = class_res;

                //cout << "Changing node " << index << " from " << previous << " to " << class_res << endl;
            }

        }

        // If any change done, stop iterating, convergence achieved!
        if ( !changes )
            keep_iterating = false;

        iteration++;
    }

    cout << "Iterations: " << iteration << endl;

    // TODO: It could be interesting return the case of stopping iterating
}


/*------------------------------------------------------------------------------

                              decodeGreedy

------------------------------------------------------------------------------*/

void CGraph::decodeGreedy( std::vector<size_t> &results )
{
    // Intilize the results vector
    size_t N_nodes = m_nodes.size();
    results.clear();
    results.resize(N_nodes);

    Eigen::VectorXd v_potentials;
    v_potentials.resize( N_nodes );

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        m_nodes[index].getPotentials().maxCoeff(&nodeMAP);
        results[index] = nodeMAP;
        v_potentials(index) = 0; //std::numeric_limits<double>::min();
    }

    // cout << "Initial_potentials: " << v_potentials << endl;

    // Set the stop conditions
    bool keep_iterating = true;

    m_TOptions.ICM_maxIterations = 1000;
    size_t iteration = 0;
    Eigen::VectorXd v_new_potentials;
    v_new_potentials.resize( N_nodes );
    vector<size_t> new_results( N_nodes );

    // Let's go!
    while ( (keep_iterating) && ( iteration < m_TOptions.ICM_maxIterations) )
    {
        bool changes = false;

        // Iterate over all the nodes, and check if a more promissing class
        // is waiting for us
        for ( size_t index = 0; index < N_nodes; index++ )
        {
            Eigen::VectorXd potentials = m_nodes[index].getPotentials();
            size_t N_neighbors = m_edges_f[index].size();

            // Iterating over the neighbors, multiplying the potential of the
            // corresponding col in the edgePotentials, according to their class
            for ( size_t neighbor = 0; neighbor < N_neighbors; neighbor++ )
            {
                size_t edgeID = m_edges_f[index][neighbor];

                size_t neighborID;
                size_t ID1, ID2;

                m_edges[edgeID].getNodesID(ID1,ID2);

                ( ID1 == index ) ? neighborID = ID2 : neighborID = ID1;


                Eigen::MatrixXd edgePotentials = m_edges[edgeID].getPotentials();

                potentials = potentials.cwiseProduct( edgePotentials.col(results[neighborID]) );
            }

            size_t class_res;
            double max_potential = potentials.maxCoeff(&class_res);

            v_new_potentials(index) = max_potential;
            new_results[index] = class_res;

        }

      //  cout << "Iteration Potentials" << endl << v_new_potentials << endl;

        Eigen::VectorXd difference = v_new_potentials - v_potentials;

       // cout << "Difference" << difference << endl;
        size_t node;
        double max_difference = difference.maxCoeff(&node);

        if ( max_difference > 0 )
        {
            v_potentials(node) = v_new_potentials(node);
            results[node] = new_results[node];
        }
        else
            keep_iterating = false;

        cout << "Iteration: " << iteration++ << endl;
    }

    // TODO: It could be interesting return the case of stopping iterating
}
