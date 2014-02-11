#include "dataTypes.h"

using namespace PGMplusplus;
using namespace std;


size_t CNode::getMostProbableClass()
{
    size_t mp_class;

    double max_value = std::numeric_limits<double>::min();

    size_t N_classes = m_potentials.rows();

    for ( size_t index = 0; index < N_classes; index++ )
    {
        if ( m_potentials[index] > max_value )
        {
            max_value = m_potentials[index];
            mp_class = index;
        }
    }

    return mp_class;
}

//----------------------------------------------------------------------------//
//                                                                            //
//                          computePotentials
//                                                                            //
//----------------------------------------------------------------------------//

void CGraph::computePotentials()
{
    // Algorithm steps:
    //  1. Compute node potentials
    //  2. Comput edge potentials

    //
    //  1. Node potentials
    //

    std::vector<CNode>::iterator it;       

    for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
    {
        // Get the node type
        size_t type = it->getType();

        // Compute the node potentials according to the node type and its
        // extracted features
        Eigen::VectorXd potentials = m_nodeWeights[type] * it->getFeatures();

        //std::cout << "Potentials: " << potentials << std::endl;

        it->setPotentials( potentials );

    }

    //std::DBL_MAX
    //  2. Edge potentials
    //

    std::vector<CEdge>::iterator it2;

    for ( it2 = m_edges.begin(); it2 != m_edges.end(); it2++ )
    {
        // Get the edge type, its extracted features and the number of them
        size_t type = it2->getType();
        Eigen::VectorXd v_feat = it2->getFeatures();
        size_t num_feat = m_edgeWeights[type].size();

        // Compute the potential for each feature, and sum up them to obtain
        // the desired edge potential
        std::vector<Eigen::MatrixXd>    potentials_per_feat(num_feat);
        Eigen::MatrixXd potentials;


        for ( size_t feat = 0; feat < num_feat; feat++ )
        {
            potentials_per_feat.at(feat) = m_edgeWeights[type][feat]*v_feat(feat);

            //std::cout << "M_potentials: " << m_edgeWeights[type][feat] << std::endl;

            if ( !feat )
                potentials = potentials_per_feat[feat];
            else
                potentials += potentials_per_feat[feat];
        }

        it2->setPotentials ( potentials );
    }

}


//----------------------------------------------------------------------------//
//                                                                            //
//                              decodeICM
//                                                                            //
//----------------------------------------------------------------------------//

void CGraph::decodeICM( std::vector<size_t> &results )
{
    // Intilize the results vector
    size_t N_nodes = m_nodes.size();
    results.clear();
    results.resize(N_nodes);

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        m_nodes[index].getPotentials().maxCoeff(&nodeMAP);
        results[index] = nodeMAP;
    }

    // Set the stop conditions
    bool keep_iterating = true;

    m_TOptions.ICM_maxIterations = 1000;
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
                size_t neighborID = m_edges[edgeID].getSecondNodeID();


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

    // TODO: It could be interesting return the case of stopping iterating
}

//----------------------------------------------------------------------------//
//                                                                            //
//                              decodeGreedy
//                                                                            //
//----------------------------------------------------------------------------//

void CGraph::decodeGreedy( std::vector<size_t> &results )
{
    // Intilize the results vector
    size_t N_nodes = m_nodes.size();
    results.clear();
    results.resize(N_nodes);

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        m_nodes[index].getPotentials().maxCoeff(&nodeMAP);
        results[index] = nodeMAP;
    }

    Eigen::VectorXd v_potentials;
    v_potentials.resize( N_nodes );

    // Compute the initial potential for each node and its neighbors according
    // to its edges
    for ( size_t index = 0; index < N_nodes; index++ )
    {
        Eigen::VectorXd potentials = m_nodes[index].getPotentials();
        size_t N_neighbors = m_edges_f[index].size();

        // Iterating over the neighbors, multiplying the potential of the
        // corresponding col in the edgePotentials, according to their class
        for ( size_t neighbor = 0; neighbor < N_neighbors; neighbor++ )
        {
            size_t edgeID = m_edges_f[index][neighbor];
            size_t neighborID = m_edges[edgeID].getSecondNodeID();


            Eigen::MatrixXd edgePotentials = m_edges[edgeID].getPotentials();

            potentials = potentials.cwiseProduct( edgePotentials.col(results[neighborID]) );
        }

        v_potentials(index) = potentials.maxCoeff();

    }

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
                size_t neighborID = m_edges[edgeID].getSecondNodeID();


                Eigen::MatrixXd edgePotentials = m_edges[edgeID].getPotentials();

                potentials = potentials.cwiseProduct( edgePotentials.col(results[neighborID]) );
            }

            //cout << "Potentials" << endl << potentials << endl;
            size_t class_res;
            double max_potential = potentials.maxCoeff(&class_res);

            v_new_potentials[index] = max_potential;
            new_results[index] = class_res;

        }

        Eigen::VectorXd difference = v_potentials - v_new_potentials;
        size_t node;
        double max_difference = difference.maxCoeff(&node);

        if ( max_difference > 0 )
        {
            v_potentials(node) = v_new_potentials(node);
            results[node] = new_results[node];
        }
        else
            keep_iterating = false;

        iteration++;
    }

    // TODO: It could be interesting return the case of stopping iterating
}
