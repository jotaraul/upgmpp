
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

                           CDecodeMaxNodePot

------------------------------------------------------------------------------*/

void CDecodeMaxNodePot::decode( CGraph &graph, std::map<size_t,size_t> &results )
{
    // Direct access to useful vbles
    const std::vector<CNodePtr> &nodes = graph.getNodes();
    size_t N_nodes = nodes.size();

    // Initialize the results vector
    results.clear();

    // Choose as class for all the nodes their more probable class
    // according to the node potentials

    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        size_t ID = nodes[index]->getID();

        nodes[index]->getPotentials( m_options.considerNodeFixedValues ).maxCoeff(&nodeMAP);
        results[ID] = nodeMAP;
    }
}


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

        nodes[index]->getPotentials( m_options.considerNodeFixedValues ).maxCoeff(&nodeMAP);
        results[ID] = nodeMAP;        
    }

    std::map<size_t,size_t>::iterator it;

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
            Eigen::VectorXd potentials = nodes[index]->getPotentials( m_options.considerNodeFixedValues );
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

        nodes[index]->getPotentials( m_options.considerNodeFixedValues ).maxCoeff(&nodeMAP);
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

            Eigen::VectorXd potentials = nodes[index]->getPotentials( m_options.considerNodeFixedValues );

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
    size_t nodeClasses = nodeType->getNumberOfClasses();
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
        VectorXd nodePotPlusIncMsg = nodePtr->getPotentials( m_options.considerNodeFixedValues );

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

/*------------------------------------------------------------------------------

                              CDecodeGraphCuts

------------------------------------------------------------------------------*/

void CDecodeGraphCuts::decode( CGraph &graph,
                         std::map<size_t,size_t> &results )
{
    cout << "Decoding graph cuts..." << endl;

    // Decoding method overview:
    // 1. Check binary states and sub-modularity conditions.
    // 2. Pepare energies and capacities.
    // 3. Solve the Max-Flow Min-Cut problem.

    //
    // 1. Check binary states and sub-modularity conditions.
    //

    // Check that all the node types have two states. This method is only for binary problems.

    vector<CNodeTypePtr> &nodeTypes= graph.getNodeTypes();
    vector<CNodeTypePtr>::iterator nodeTypesIt;

    for ( nodeTypesIt = nodeTypes.begin(); nodeTypesIt != nodeTypes.end(); nodeTypesIt++ )
    {        
        if ( (*nodeTypesIt)->getNumberOfClasses() != 2 )
        {
            cout << "[ERROR] The number of classes of a node type used in the graph is not 2." << endl;
            return;
        }
    }

    // Check that all the edge energies fulfill the sub-modularity condition

    vector<CEdgePtr> &v_edges = graph.getEdges();
    vector<CEdgePtr>::const_iterator edgesIt;
    double epsilon = exp(-1);//Eigen::NumTraits<double>::epsilon();


    for ( edgesIt = v_edges.begin(); edgesIt != v_edges.end(); edgesIt++ )
    {
        MatrixXd potentials = (*edgesIt)->getPotentials();

        if ( (potentials.cols() != 2) || ( potentials.rows() != 2 ) )
        {
            cout << "[ERROR] The number of classes of a node into an edge is not two." << endl;
            return;
        }

        MatrixXd energies = -(potentials.array().log());

        //cout << "Potentials: " << endl << potentials << endl;
        //cout << "Energies: " << endl << energies << endl;

        if ( energies(0,0) + energies(1,1) > energies(0,1) + energies(1,0) + exp(-15) )
        {
            cout << "[ERROR] Implemented graph cut method only valid for sub-modular potentials." << endl;
            cout <<  "energies(0,0) + energies(1,1)=" << energies(0,0) + energies(1,1);
            cout << " and energies(0,1) + energies(1,0)=" << energies(0,1) + energies(1,0) << endl;
            return;
        }
    }

    // Make node energies
    map<size_t,Vector2d> nodeEnergies;
    map<size_t,size_t>   nodesMap;

    vector<CNodePtr> &v_nodes = graph.getNodes();
    vector<CNodePtr>::const_iterator nodesIt;
    size_t index = 1;

    for ( nodesIt = v_nodes.begin(); nodesIt != v_nodes.end(); nodesIt++ )
    {
        size_t ID           = (*nodesIt)->getID();
        Vector2d potentials = (*nodesIt)->getPotentials();
        Vector2d energies   = -(potentials.array().log());

        nodeEnergies[ID] = energies;
        nodesMap[ID] = index++;

        //cout << "Energies of node " << (*nodesIt)->getID() << endl << energies << endl;
    }


    /*for ( edgesIt = v_edges.begin(); edgesIt != v_edges.end(); edgesIt++ )
    {
        MatrixXd potentials = (*edgesIt)->getPotentials();
        size_t ID1, ID2;
        (*edgesIt)->getNodesID(ID1,ID2);

        MatrixXd energies = -(potentials.array().log());

        nodeEnergies[ID1](1) = nodeEnergies[ID1](1) + energies(1,0) - energies(0,0);
        nodeEnergies[ID2](1) = nodeEnergies[ID2](1) + energies(1,1) - energies(1,0);

        //cout << "Energies of node " << ID1 << endl << nodeEnergies[ID1] << endl;
        //cout << "Energies of node " << ID2 << endl << nodeEnergies[ID2] << endl;
    }*/

    size_t N_nodes = v_nodes.size() + 2; // Number of nodes in the graph plus source plus tap

    MatrixXd capacities(N_nodes,N_nodes);
    capacities.setZero();

    // Fill the capacities between nodes in the graph

    for ( edgesIt = v_edges.begin(); edgesIt != v_edges.end(); edgesIt++ )
    {
        MatrixXd potentials = (*edgesIt)->getPotentials();
        MatrixXd energies = -(potentials.array().log());

        //cout << "Energies of edge " << (*edgesIt)->getID() << endl << energies << endl;

        size_t ID1, ID2;
        (*edgesIt)->getNodesID(ID1,ID2);

        //cout << "Setting capacities of: " << nodesMap[ID1] << " " << nodesMap[ID2]
        //     << " to " << energies(0,1) + energies(1,0) - energies(0,0) - energies(1,1) << endl;
        capacities(nodesMap[ID1],nodesMap[ID2]) =
        //capacities(nodesMap[ID2],nodesMap[ID1]) =
                energies(0,1) + energies(1,0) - energies(0,0) - energies(1,1);

        /*if (capacities(nodesMap[ID1],nodesMap[ID2]) < exp(-10) )
                capacities(nodesMap[ID1],nodesMap[ID2]) = 0;*/
    }

    // Fill the capacities between nodes and soruce/tap
    /*
    for ( nodesIt = v_nodes.begin(); nodesIt != v_nodes.end(); nodesIt++ )
    {
        size_t ID = (*nodesIt)->getID();

        cout << "Jar: " <<  nodeEnergies[ID](1) - nodeEnergies[ID](0) << endl;

        if ( nodeEnergies[ID](0) < nodeEnergies[ID](1) )
            capacities(0,nodesMap[ID]) = nodeEnergies[ID](1) - nodeEnergies[ID](0);
        else
            capacities(N_nodes-1,nodesMap[ID]) = nodeEnergies[ID](0) - nodeEnergies[ID](1);
    }*/

    for ( nodesIt = v_nodes.begin(); nodesIt != v_nodes.end(); nodesIt++ )
    {
        size_t ID = (*nodesIt)->getID();

        //cout << "Jar: " <<  nodeEnergies[ID](1) - nodeEnergies[ID](0) << endl;

        if ( nodeEnergies[ID](0) < nodeEnergies[ID](1) )
            capacities(0,nodesMap[ID]) = nodeEnergies[ID](1) - nodeEnergies[ID](0);
        else
            capacities(nodesMap[ID],N_nodes-1) = nodeEnergies[ID](0) - nodeEnergies[ID](1);
    }

    for ( edgesIt = v_edges.begin(); edgesIt != v_edges.end(); edgesIt++ )
    {
        MatrixXd potentials = (*edgesIt)->getPotentials();
        size_t ID1, ID2;
        (*edgesIt)->getNodesID(ID1,ID2);

        MatrixXd energies = -(potentials.array().log());

        if ( energies(1,0) - energies(0,0) > 0)
            capacities(0,nodesMap[ID1]) += energies(1,0) - energies(0,0);
        else
            capacities(nodesMap[ID1],N_nodes-1) += - (energies(1,0) - energies(0,0));

        if ( energies(1,1) - energies(1,0) > 0 )
            capacities(0,nodesMap[ID2]) += energies(1,1) - energies(1,0);
        else
            capacities(nodesMap[ID2],N_nodes-1) += - (energies(1,1) - energies(1,0));

        //cout << "Energies of node " << ID1 << endl << nodeEnergies[ID1] << endl;
        //cout << "Energies of node " << ID2 << endl << nodeEnergies[ID2] << endl;
    }

    // Solve the Max-Flow Min-Cut problem

    VectorXi cut(N_nodes);
    cut.setZero();

    //cout << "Capacities: " << endl << capacities << endl;


    fordFulkerson(capacities, 0, N_nodes-1, cut);

    for ( nodesIt = v_nodes.begin(); nodesIt != v_nodes.end(); nodesIt++ )
    {
        size_t ID           = (*nodesIt)->getID();
        results[ID] = (cut[nodesMap[ID]]==1) ? 0 : 1;        
    }
}

/*------------------------------------------------------------------------------

                            CDecodeAlphaExpansion

------------------------------------------------------------------------------*/

void CDecodeAlphaExpansion::decode( CGraph &graph,
                                    std::map<size_t,size_t> &results )
{

    //
    // Method workflow:
    // 1. Compute the initial assignation to variables (the classes with the
    //    higher node potential).
    // 2. Do Aplha-expansions until convergence or a given number of iterations
    //    is reached.
    //      2.1 For each nodeType, do alpha-expansions
    //          2.1.1 Get a bound graph where nodes with the current state (alpha)
    //                are bound to @state.
    //          2.1.2 Binarize the bound graph, so only two states are possible:
    //                alpha, and the previous one from the node.
    //          2.1.3 Compute graph cuts decoding on the resultant graph.
    //          2.1.4 For each node assigned to @state in the decoding result,
    //                do an alpha move.
    //      2.2 Check convergency.
    //

    cout << "Decoding Alpha expansion..." << endl;

    // Initialize the results vector
    results.clear();

    // Direct access to useful vbles

    const std::vector<CNodePtr> &nodes  = graph.getNodes();
    size_t N_nodes                      = nodes.size();

    //
    // 1. Initial assignation to vbles
    //

    std::map<size_t,size_t> assignation;
    std::map<size_t,size_t> assignation_old;

    // Choose as initial class for all the nodes their more probable class
    // according to the node potentials

    for ( size_t index = 0; index < N_nodes; index++ )
    {
        size_t nodeMAP;
        size_t ID = nodes[index]->getID();

        nodes[index]->getPotentials( m_options.considerNodeFixedValues ).maxCoeff(&nodeMAP);
        assignation[ID] = nodeMAP;
    }

    // Initial assignation
    std::map<size_t,size_t>::iterator it;

    for ( it = assignation.begin(); it != assignation.end(); it++ )
    {
        std::cout << "Node id " << it->first << " labeled as " << it->second << std::endl;
    }

    // Get the likelihood of this assignation. Useful for convergence checking
    double totalPotential = graph.getUnnormalizedLogLikelihood( assignation );

    //
    // 2. Do Alpha-expansions until convergence or a given number of iterations is reached.
    //

    bool convergence = false;
    size_t iteration = 0;

    while ( !convergence && ( iteration < m_options.maxIterations ) )
    {
        // Store the previous assignation for convergence checking
        assignation_old = assignation;

        // Iterate over the nodeTypes while moving stuff!
        vector<CNodeTypePtr> &nodeTypes = graph.getNodeTypes();
        vector<CNodeTypePtr>::const_iterator itNodeTypes;

        for ( itNodeTypes = nodeTypes.begin(); itNodeTypes != nodeTypes.end(); itNodeTypes++)
        {
            size_t nodeTypeID = (*itNodeTypes)->getID();            
            size_t N_classes  = (*itNodeTypes)->getNumberOfClasses();

            //
            // 2.1 Ok, move across all the possible classes/states of current node type
            // For each state, check the nodes which assignation is different to
            // such a state (alpha), and check if a move to alpha is nice.
            //

            //for ( size_t state = 0; state < N_classes; state++ )
            for ( size_t state = 0; state < N_classes; state++ )
            {
                cout << "Expanding state " << state << " in nodes of type " << nodeTypeID << endl;

                map<size_t,size_t>  nodesToBind;
                size_t              nodesOfThisType = 0;

                // Iterate over all the nodes in the graph
                for ( size_t node = 0; node < nodes.size(); node++ )
                {
                    // Checking if they share the type with the current NodeType
                    if ( nodes[node]->getType()->getID() == nodeTypeID )
                    {
                        // And if their assigned value is the same as the current state.
                        // This nodes will be bound to @state, so the resultant graph only
                        // has nodes with assignation different to @state (alpha).
                        if ( assignation[nodes[node]->getID()] == state )
                            nodesToBind[nodes[node]->getID()] = state;

                        nodesOfThisType++;
                    }
                }

                //
                // 2.1.1 Get a bound graph where nodes with the current state
                //       (alpha) are bound to @state.
                //

                CGraph boundGraph;

                if ( nodesToBind.size() == nodesOfThisType ) // Nothing to move
                    continue;
                else // Bind
                    graph.getBoundGraph( boundGraph, nodesToBind );

                //cout << "Graph:" << endl << graph << endl;
                //cout << "Bound Graph:" << endl << boundGraph << endl;

                //
                // Prepare the problem so it can be affordable using graph cuts
                //

                //
                // 2.1.2 Binarize the bound graph, so only two states are possible:
                //       alpha, and the previous one from the node.
                //

                CNodeTypePtr binaryNodeTypePtr( new CNodeType(2,1) );
                CEdgeTypePtr binaryEdgeTypePtr( new CEdgeType(1,binaryNodeTypePtr,
                                                              binaryNodeTypePtr) );

                vector<CNodePtr> &boundNodes = boundGraph.getNodes();

                for  (size_t node = 0; node < boundNodes.size(); node++ )
                {
                    VectorXd potentials = boundNodes[node]->getPotentials();
                    size_t nodeID     =   boundNodes[node]->getID();
                    size_t currentNodeTypeID =   boundNodes[node]->getType()->getID();

                    double previousClassPotential = potentials( assignation[nodeID] );
                    double alphaClassPotential;

                    // Check if the nodeType matches
                    if ( currentNodeTypeID == nodeTypeID )
                        alphaClassPotential = potentials( state );
                    else // if not, the state of the node wont change
                        alphaClassPotential = 0;

                    Vector2d binaryPotentials;

                    // First, the potential for the alpha state.
                    // Second, the potential for the previous node assignation.
                    binaryPotentials << alphaClassPotential, previousClassPotential;

                    boundGraph.setNodeType(boundNodes[node],binaryNodeTypePtr);
                    boundNodes[node]->setPotentials( binaryPotentials );

                }

                // ... and binary edges

                vector<CEdgePtr> &boundEdges = boundGraph.getEdges();

                for ( size_t edge = 0; edge < boundEdges.size(); edge++ )
                {
                    CNodePtr nodePtr1, nodePtr2;
                    boundEdges[edge]->getNodes(nodePtr1,nodePtr2);
                    MatrixXd &potentials = boundEdges[edge]->getPotentials();

                    MatrixXd binaryPotentials;
                    binaryPotentials.resize(2,2);

                    if ( ( nodePtr1->getType()->getID() == nodeTypeID) &&
                         ( nodePtr2->getType()->getID() == nodeTypeID ) )
                    {
                        binaryPotentials(0,0) = potentials( state, state );
                        binaryPotentials(0,1) = potentials( state, assignation[nodePtr2->getID()] );
                        binaryPotentials(1,0) = potentials( assignation[nodePtr1->getID()], state );
                        binaryPotentials(1,1) = potentials( assignation[nodePtr1->getID()],
                                                            assignation[nodePtr2->getID()] );

                    }
                    else // Different node types
                    {
                        if ( nodePtr1->getType()->getID() != nodeTypeID)
                        {
                            binaryPotentials(0,0) = 0;
                            binaryPotentials(0,1) = 0;
                            binaryPotentials(1,0) = potentials( assignation[nodePtr1->getID()], state );
                            binaryPotentials(1,1) = potentials( assignation[nodePtr1->getID()],
                                                                assignation[nodePtr2->getID()] );
                        }
                        else // ( nodePtr2->getType()->getID() != nodeTypeID)
                        {
                            binaryPotentials(0,0) = 0;
                            binaryPotentials(0,1) = potentials( state, assignation[nodePtr2->getID()] );
                            binaryPotentials(1,0) = 0;
                            binaryPotentials(1,1) = potentials( assignation[nodePtr1->getID()],
                                                                assignation[nodePtr2->getID()] );
                        }

                    }

                    //
                    // Tune edges so they fulfill the sub-modular condition
                    //

                    Matrix2d binaryEnergies = -(binaryPotentials.array().log());
                    double aux = binaryEnergies(0,1)+binaryEnergies(1,0)-binaryEnergies(0,0);

                    //cout << "Aux: " << aux << endl << "BinaryEnergies(1,1): " << binaryEnergies(1,1) << endl;
                    if ( binaryEnergies(1,1) > aux )
                        binaryEnergies(1,1) = aux;

                    binaryPotentials = (-binaryEnergies).array().exp();

                    boundEdges[edge]->setType( binaryEdgeTypePtr );
                    boundEdges[edge]->setPotentials( binaryPotentials );

                }

                //cout << "Binary Graph:" << endl << boundGraph << endl;

                //
                // Call to the decodeGraphCuts method with the bound and tunned graph
                //

                CDecodeGraphCuts decodeGraphCuts;
                CDecodeICMGreedy decodeICMGreedy;
                map<size_t,size_t> gcResults;

                //decodeGraphCuts.decode( boundGraph, gcResults );
                decodeICMGreedy.decode( boundGraph, gcResults );

                for ( map<size_t,size_t>::iterator gcResIt = gcResults.begin(); gcResIt!= gcResults.end(); gcResIt++)
                {
                    if ( gcResIt->second == 0 )
                    {
                        assignation[ gcResIt->first ] = state;
                    }
                }

            }
        }

        //
        // 2.2 Check termination (convergence) conditions
        //

        if ( assignation == assignation_old) // Same assignation
        {
            cout << "Convergence achieved: the same assignation" << endl;
            convergence = true;
            continue;
        }

        double newTotalPotential = graph.getUnnormalizedLogLikelihood( assignation );

        if ( newTotalPotential = totalPotential ) // Same likelihood
        {
            cout << "Convergence achieved: the same likelihood" << endl;
            continue;
        }
        else
            totalPotential = newTotalPotential;

        iteration++;
    }

    results = assignation;
}

