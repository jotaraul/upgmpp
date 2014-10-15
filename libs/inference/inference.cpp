
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

#include "inference.hpp"

using namespace UPGMpp;
using namespace std;
using namespace Eigen;



/*------------------------------------------------------------------------------

                               CLBPInference

------------------------------------------------------------------------------*/

void CLBPInference::infer(CGraph &graph,
                          map<size_t,VectorXd> &nodeBeliefs,
                          map<size_t,MatrixXd> &edgeBeliefs,
                          double &logZ)
{
    //
    //  Algorithm workflow:
    //  1. Compute the messages passed
    //  2. Compute node beliefs
    //  3. Compute edge beliefs
    //  4. Compute logZ
    //

    nodeBeliefs.clear();
    edgeBeliefs.clear();

    const vector<CNodePtr> nodes = graph.getNodes();
    const vector<CEdgePtr> edges = graph.getEdges();
    multimap<size_t,CEdgePtr> edges_f = graph.getEdgesF();

    size_t N_nodes = nodes.size();
    size_t N_edges = edges.size();

    //
    // 1. Compute the messages passed
    //

    vector<vector<VectorXd> >   messages;
    bool                        maximize = false;

    messagesLBP( graph, m_options, messages, maximize );

    //
    // 2. Compute node beliefs
    //

    for ( size_t nodeIndex = 0; nodeIndex < N_nodes; nodeIndex++ )
    {
        const CNodePtr nodePtr = graph.getNode( nodeIndex );
        size_t         nodeID  = nodePtr->getID();
        VectorXd       nodePotPlusIncMsg = nodePtr->getPotentials( m_options.considerNodeFixedValues );

        NEIGHBORS_IT neighbors = edges_f.equal_range(nodeID);

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

        nodeBeliefs[ nodeID ] = nodePotPlusIncMsg;

        //cout << "Beliefs of node " << nodeIndex << endl << nodePotPlusIncMsg << endl;
    }

    //
    // 3. Compute edge beliefs
    //

    for ( size_t edgeIndex = 0; edgeIndex < N_edges; edgeIndex++ )
    {
        CEdgePtr edgePtr = edges[edgeIndex];
        size_t   edgeID  = edgePtr->getID();

        size_t ID1, ID2;
        edgePtr->getNodesID( ID1, ID2 );

        MatrixXd edgePotentials = edgePtr->getPotentials();
        MatrixXd edgeBelief = edgePotentials;

        VectorXd &message1To2 = messages[edgeIndex][0];
        VectorXd &message2To1 = messages[edgeIndex][1];

        //cout << "----------------------" << endl;
        //cout << nodeBeliefs[ ID1 ] << endl;
        //cout << "----------------------" << endl;
        //cout << message2To1 << endl;

        VectorXd node1Belief = nodeBeliefs[ ID1 ].cwiseQuotient( message2To1 );
        VectorXd node2Belief = nodeBeliefs[ ID2 ].cwiseQuotient( message1To2 );

        //cout << "----------------------" << endl;

        MatrixXd node1BeliefMatrix ( edgePotentials.rows(), edgePotentials.cols() );
        for ( size_t row = 0; row < edgePotentials.rows(); row++ )
            for ( size_t col = 0; col < edgePotentials.cols(); col++ )
                node1BeliefMatrix(row,col) = node1Belief(row);

        //cout << "Node 1 belief matrix: " << endl << node1BeliefMatrix << endl;

        edgeBelief = edgeBelief.cwiseProduct( node1BeliefMatrix );

        MatrixXd node2BeliefMatrix ( edgePotentials.rows(), edgePotentials.cols() );
        for ( size_t row = 0; row < edgePotentials.rows(); row++ )
            for ( size_t col = 0; col < edgePotentials.cols(); col++ )
                node2BeliefMatrix(row,col) = node2Belief(col);

        //cout << "Node 2 belief matrix: " << endl << node2BeliefMatrix << endl;

        edgeBelief = edgeBelief.cwiseProduct( node2BeliefMatrix );

        //cout << "Edge potentials" << endl << edgePotentials << endl;
        //cout << "Edge beliefs" << endl << edgeBelief << endl;

        // Normalize
        edgeBelief = edgeBelief / edgeBelief.sum();



        edgeBeliefs[ edgeID ] = edgeBelief;
    }

    //
    // 4. Compute logZ
    //

    double energyNodes  = 0;
    double energyEdges  = 0;
    double entropyNodes = 0;
    double entropyEdges = 0;

    // Compute energy and entropy from nodes

    for ( size_t nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++ )
    {
        CNodePtr nodePtr     = nodes[ nodeIndex ];
        size_t   nodeID      = nodePtr->getID();
        size_t   N_Neighbors = graph.getNumberOfNodeNeighbors( nodeID );

        // Useful computations and shorcuts
        VectorXd &nodeBelief        = nodeBeliefs[nodeID];
        VectorXd logNodeBelief      = nodeBeliefs[nodeID].array().log();
        VectorXd nodePotentials    = nodePtr->getPotentials( m_options.considerNodeFixedValues );
        VectorXd logNodePotentials = nodePotentials.array().log();

        // Entropy from the node
        energyNodes += N_Neighbors*( nodeBelief.cwiseProduct( logNodeBelief ).sum() );

        // Energy from the node
        entropyNodes += N_Neighbors*( nodeBelief.cwiseProduct( logNodePotentials ).sum() );
    }

    // Compute energy and entropy from nodes

    for ( size_t edgeIndex = 0; edgeIndex < N_edges; edgeIndex++ )
    {
        CEdgePtr edgePtr = edges[ edgeIndex ];
        size_t   edgeID  = edgePtr->getID();

        // Useful computations and shorcuts
        MatrixXd &edgeBelief       = edgeBeliefs[ edgeID ];
        MatrixXd logEdgeBelief     = edgeBelief.array().log();
        MatrixXd &edgePotentials   = edgePtr->getPotentials();
        MatrixXd logEdgePotentials = edgePotentials.array().log();

        // Entropy from the edge
        energyEdges += edgeBelief.cwiseProduct( logEdgeBelief ).sum();

        // Energy from the edge
        entropyEdges += edgeBelief.cwiseProduct( logEdgePotentials ).sum();

    }

    // Final Bethe free energy

    double BethefreeEnergy = ( energyNodes - energyEdges ) - ( entropyNodes - entropyEdges );

    // Compute logZ

    logZ = - BethefreeEnergy;

}


/*------------------------------------------------------------------------------

                               CTRPBPInference

------------------------------------------------------------------------------*/

void CTRPBPInference::infer(CGraph &graph,
                          map<size_t,VectorXd> &nodeBeliefs,
                          map<size_t,MatrixXd> &edgeBeliefs,
                          double &logZ)
{
    //
    //  Algorithm workflow:
    //  1. Compute the messages passed
    //  2. Compute node beliefs
    //  3. Compute edge beliefs
    //  4. Compute logZ
    //

    nodeBeliefs.clear();
    edgeBeliefs.clear();

    const vector<CNodePtr> nodes = graph.getNodes();
    const vector<CEdgePtr> edges = graph.getEdges();
    multimap<size_t,CEdgePtr> edges_f = graph.getEdgesF();

    size_t N_nodes = nodes.size();
    size_t N_edges = edges.size();

    //
    // 1. Create spanning trees
    //

    bool allNodesAdded = false;
    vector<vector<size_t > > v_trees;
    vector<bool> v_addedNodes(N_nodes,false);
    map<size_t,size_t> addedNodesMap;

    for (size_t i = 0; i < N_nodes; i++)
        addedNodesMap[ nodes[i]->getID() ] = i;

    while (!allNodesAdded)
    {
        allNodesAdded = true;

        vector<size_t> tree;
        getSpanningTree( graph, tree );

        // Check that the tree is not empty
        if ( tree.size() )
            v_trees.push_back( tree );

        cout << "Tree: ";

        for ( size_t i_node = 0; i_node < tree.size(); i_node++ )
        {
            v_addedNodes[ addedNodesMap[tree[i_node]] ] = true;
            cout << tree[i_node] << " ";
        }

        cout << endl;

        for ( size_t i_node = 0; i_node < N_nodes; i_node++ )
            if ( !v_addedNodes[i_node] )
            {
                allNodesAdded = false;
                break;
            }

    }


    //
    // 1. Compute messages passed in each tree until convergence
    //

    vector<vector<VectorXd> >   messages;
    bool                        maximize = false;

    double totalSumOfMsgs = std::numeric_limits<double>::max();

    size_t iteration;

    for ( iteration = 0; iteration < m_options.maxIterations; iteration++ )
    {

        for ( size_t i_tree=0; i_tree < v_trees.size(); i_tree++ )
            messagesLBP( graph, m_options, messages, maximize, v_trees[i_tree] );

        double newTotalSumOfMsgs = 0;
        for ( size_t i = 0; i < N_edges; i++ )
        {
            newTotalSumOfMsgs += messages[i][0].sum() + messages[i][1].sum();
        }

        if ( std::abs( totalSumOfMsgs - newTotalSumOfMsgs ) <
             m_options.convergency )
            break;

        totalSumOfMsgs = newTotalSumOfMsgs;

    }

    //
    // 2. Compute node beliefs
    //

    for ( size_t nodeIndex = 0; nodeIndex < N_nodes; nodeIndex++ )
    {
        const CNodePtr nodePtr = graph.getNode( nodeIndex );
        size_t         nodeID  = nodePtr->getID();
        VectorXd       nodePotPlusIncMsg = nodePtr->getPotentials( m_options.considerNodeFixedValues );

        NEIGHBORS_IT neighbors = edges_f.equal_range(nodeID);

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

        nodeBeliefs[ nodeID ] = nodePotPlusIncMsg;

        //cout << "Beliefs of node " << nodeIndex << endl << nodePotPlusIncMsg << endl;
    }

    //
    // 3. Compute edge beliefs
    //

    for ( size_t edgeIndex = 0; edgeIndex < N_edges; edgeIndex++ )
    {
        CEdgePtr edgePtr = edges[edgeIndex];
        size_t   edgeID  = edgePtr->getID();

        size_t ID1, ID2;
        edgePtr->getNodesID( ID1, ID2 );

        MatrixXd edgePotentials = edgePtr->getPotentials();
        MatrixXd edgeBelief = edgePotentials;

        VectorXd &message1To2 = messages[edgeIndex][0];
        VectorXd &message2To1 = messages[edgeIndex][1];

        //cout << "----------------------" << endl;
        //cout << nodeBeliefs[ ID1 ] << endl;
        //cout << "----------------------" << endl;
        //cout << message2To1 << endl;

        VectorXd node1Belief = nodeBeliefs[ ID1 ].cwiseQuotient( message2To1 );
        VectorXd node2Belief = nodeBeliefs[ ID2 ].cwiseQuotient( message1To2 );

        //cout << "----------------------" << endl;

        MatrixXd node1BeliefMatrix ( edgePotentials.rows(), edgePotentials.cols() );
        for ( size_t row = 0; row < edgePotentials.rows(); row++ )
            for ( size_t col = 0; col < edgePotentials.cols(); col++ )
                node1BeliefMatrix(row,col) = node1Belief(row);

        //cout << "Node 1 belief matrix: " << endl << node1BeliefMatrix << endl;

        edgeBelief = edgeBelief.cwiseProduct( node1BeliefMatrix );

        MatrixXd node2BeliefMatrix ( edgePotentials.rows(), edgePotentials.cols() );
        for ( size_t row = 0; row < edgePotentials.rows(); row++ )
            for ( size_t col = 0; col < edgePotentials.cols(); col++ )
                node2BeliefMatrix(row,col) = node2Belief(col);

        //cout << "Node 2 belief matrix: " << endl << node2BeliefMatrix << endl;

        edgeBelief = edgeBelief.cwiseProduct( node2BeliefMatrix );

        //cout << "Edge potentials" << endl << edgePotentials << endl;
        //cout << "Edge beliefs" << endl << edgeBelief << endl;

        // Normalize
        edgeBelief = edgeBelief / edgeBelief.sum();



        edgeBeliefs[ edgeID ] = edgeBelief;
    }

    //
    // 4. Compute logZ
    //

    double energyNodes  = 0;
    double energyEdges  = 0;
    double entropyNodes = 0;
    double entropyEdges = 0;

    // Compute energy and entropy from nodes

    for ( size_t nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++ )
    {
        CNodePtr nodePtr     = nodes[ nodeIndex ];
        size_t   nodeID      = nodePtr->getID();
        size_t   N_Neighbors = graph.getNumberOfNodeNeighbors( nodeID );

        // Useful computations and shorcuts
        VectorXd &nodeBelief        = nodeBeliefs[nodeID];
        VectorXd logNodeBelief      = nodeBeliefs[nodeID].array().log();
        VectorXd nodePotentials    = nodePtr->getPotentials( m_options.considerNodeFixedValues );
        VectorXd logNodePotentials = nodePotentials.array().log();

        // Entropy from the node
        energyNodes += N_Neighbors*( nodeBelief.cwiseProduct( logNodeBelief ).sum() );

        // Energy from the node
        entropyNodes += N_Neighbors*( nodeBelief.cwiseProduct( logNodePotentials ).sum() );
    }

    // Compute energy and entropy from nodes

    for ( size_t edgeIndex = 0; edgeIndex < N_edges; edgeIndex++ )
    {
        CEdgePtr edgePtr = edges[ edgeIndex ];
        size_t   edgeID  = edgePtr->getID();

        // Useful computations and shorcuts
        MatrixXd &edgeBelief       = edgeBeliefs[ edgeID ];
        MatrixXd logEdgeBelief     = edgeBelief.array().log();
        MatrixXd &edgePotentials   = edgePtr->getPotentials();
        MatrixXd logEdgePotentials = edgePotentials.array().log();

        // Entropy from the edge
        energyEdges += edgeBelief.cwiseProduct( logEdgeBelief ).sum();

        // Energy from the edge
        entropyEdges += edgeBelief.cwiseProduct( logEdgePotentials ).sum();

    }

    // Final Bethe free energy

    double BethefreeEnergy = ( energyNodes - energyEdges ) - ( entropyNodes - entropyEdges );

    // Compute logZ

    logZ = - BethefreeEnergy;

}

/*------------------------------------------------------------------------------

                               CRBPInference

------------------------------------------------------------------------------*/

void CRBPInference::infer(CGraph &graph,
                          map<size_t,VectorXd> &nodeBeliefs,
                          map<size_t,MatrixXd> &edgeBeliefs,
                          double &logZ)
{
    CLBPInference LBPinference;

    m_options.particularS["order"] = "RBP";

    LBPinference.setOptions( m_options );
    LBPinference.infer( graph, nodeBeliefs, edgeBeliefs, logZ );

}
