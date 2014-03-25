

#include "inference_utils.hpp"

#include <vector>

using namespace UPGMpp;
using namespace std;
using namespace Eigen;

size_t UPGMpp::messagesLBP(CGraph &graph,
                            TInferenceOptions &options,
                            vector<vector<VectorXd> > &messages ,
                            bool maximize )
{
    const vector<CNodePtr> nodes = graph.getNodes();
    const vector<CEdgePtr> edges = graph.getEdges();
    multimap<size_t,CEdgePtr> edges_f = graph.getEdgesF();

    size_t N_nodes = nodes.size();
    size_t N_edges = edges.size();

    graph.computePotentials();

    //
    // Build the messages structure
    //

    double totalSumOfMsgs = 0;

    messages.resize( N_edges);

    for ( size_t i = 0; i < N_edges; i++ )
    {
        messages[i].resize(2);

        size_t ID1, ID2;
        edges[i]->getNodesID(ID1,ID2);

        // Messages from first node of the edge to the second one, so the size of
        // the message has to be the same as the number of classes of the second node.
        messages[i][0].resize( graph.getNodeWithID( ID2 )->getPotentials( options.considerNodeFixedValues ).rows() );
        messages[i][0].fill(1);
        // Just the opposite as before.
        messages[i][1].resize( graph.getNodeWithID( ID1 )->getPotentials( options.considerNodeFixedValues ).rows() );
        messages[i][1].fill(1);

        totalSumOfMsgs += messages[i][0].rows() + messages[i][1].rows();

    }

    //
    // Iterate until convergence or a certain maximum number of iterations is reached
    //

    size_t iteration;

    for ( iteration = 0; iteration < options.maxIterations; iteration++ )
    {
        //
        // Iterate over all the nodes
        //
        for ( size_t nodeIndex = 0; nodeIndex < N_nodes; nodeIndex++ )
        {
            const CNodePtr nodePtr = graph.getNode( nodeIndex );
            size_t nodeID          = nodePtr->getID();

            pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator > neighbors;

            neighbors = edges_f.equal_range(nodeID);

            //
            // Send a message to each neighbor
            //
            for ( multimap<size_t,CEdgePtr>::iterator itNeigbhor = neighbors.first;
                  itNeigbhor != neighbors.second;
                  itNeigbhor++ )
            {
                VectorXd nodePotPlusIncMsg = nodePtr->getPotentials( options.considerNodeFixedValues);
                size_t neighborID;
                size_t ID1, ID2;
                CEdgePtr edgePtr( (*itNeigbhor).second );
                edgePtr->getNodesID(ID1,ID2);
                ( ID1 == nodeID ) ? neighborID = ID2 : neighborID = ID1;

                //
                // Compute the message from current node as a product of all the
                // incoming messages less the one from the current neighbor
                // plus the node potential of the current node.
                //
                for ( multimap<size_t,CEdgePtr>::iterator itNeigbhor2 = neighbors.first;
                      itNeigbhor2 != neighbors.second;
                      itNeigbhor2++ )
                {
                    size_t ID11, ID12;
                    CEdgePtr edgePtr2( (*itNeigbhor2).second );
                    edgePtr2->getNodesID(ID11,ID12);
                    size_t edgeIndex = graph.getEdgeIndex( edgePtr2->getID() );

                    // Check if the current neighbor appears in the edge
                    if ( ( neighborID != ID11 ) && ( neighborID != ID12 ) )
                    {
                        if ( nodeID == ID11 )
                            nodePotPlusIncMsg = nodePotPlusIncMsg.cwiseProduct(messages[ edgeIndex ][ 1 ]);
                        else // nodeID == ID2
                            nodePotPlusIncMsg = nodePotPlusIncMsg.cwiseProduct(messages[ edgeIndex ][ 0 ]);
                    }
                }

                //cout << "Node pot" << nodePotPlusIncMsg << endl;

                //
                // Take also the potential between the two nodes
                //
                MatrixXd edgePotentials;

                if ( nodeID != ID1 )
                    edgePotentials = edgePtr->getPotentials();
                else
                    edgePotentials = edgePtr->getPotentials().transpose();

                VectorXd newMessage;
                size_t edgeIndex = graph.getEdgeIndex( edgePtr->getID() );

                if ( !maximize )
                {
                    // Multiply both, and update the potential

                    newMessage = edgePotentials * nodePotPlusIncMsg;
                }
                else
                {

                    if ( nodeID == ID1 )
                        newMessage.resize(messages[ edgeIndex ][0].rows());
                    else
                        newMessage.resize(messages[ edgeIndex ][1].rows());

                    for ( size_t row = 0; row < edgePotentials.rows(); row++ )
                    {
                        double maxRowValue = std::numeric_limits<double>::min();

                        for ( size_t col = 0; col < edgePotentials.cols(); col++ )
                        {
                            double value = edgePotentials(row,col)*nodePotPlusIncMsg(col);
                            if ( value > maxRowValue )
                                maxRowValue = value;
                        }
                        newMessage(row) = maxRowValue;
                    }

                    // Normalize new message
                    newMessage = newMessage / newMessage.sum();

                    //cout << "New message: " << endl << newMessage << endl;
                }

                //
                // Set the message!
                //

                if ( nodeID == ID1 )
                    messages[ edgeIndex ][0] = newMessage;
                else
                    messages[ edgeIndex ][1] = newMessage;

            }

        } // Nodes

        //
        // Check convergency!!
        //

        double newTotalSumOfMsgs = 0;
        for ( size_t i = 0; i < N_edges; i++ )
        {
            newTotalSumOfMsgs += messages[i][0].sum() + messages[i][1].sum();
        }

        //printf("%4.10f\n",std::abs( totalSumOfMsgs - newTotalSumOfMsgs ));

        if ( std::abs( totalSumOfMsgs - newTotalSumOfMsgs ) <
             options.convergency )
            break;

        totalSumOfMsgs = newTotalSumOfMsgs;

    } // Iterations

    return 1;
}
