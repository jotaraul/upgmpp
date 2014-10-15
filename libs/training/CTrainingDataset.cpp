
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

#include "CTrainingDataset.hpp"
#include <lbfgs.h>
#include "inference.hpp"
#include "decoding.hpp"

#include <vector>

using namespace std;
using namespace UPGMpp;


/*------------------------------------------------------------------------------

                             updateNodeTypeWeights

------------------------------------------------------------------------------*/

void updateNodeTypeWeights( Eigen::MatrixXi & nodeWeightsMap,
                        CNodeTypePtr nodeType,
                        const double *x )
{
    Eigen::MatrixXd &weights = nodeType->getWeights();

    size_t N_cols = weights.cols();
    size_t N_rows = weights.rows();

    for ( size_t row = 0; row < N_rows; row++ )
        for ( size_t col = 0; col < N_cols; col++ )
        {
            weights(row,col) = x[nodeWeightsMap(row,col)];
        }
}


/*------------------------------------------------------------------------------

                              updateEdgeTypeWeights

------------------------------------------------------------------------------*/

void updateEdgeTypeWeights( std::vector<Eigen::MatrixXi>& edgeWeightsMap,
                        CEdgeTypePtr  edgeType,
                        const double *x)
{
    std::vector<Eigen::MatrixXd> &weights = edgeType->getWeights();
    for ( size_t feature = 0; feature < weights.size(); feature++ )
    {
        size_t N_cols = weights[feature].cols();
        size_t N_rows = weights[feature].rows();

        for ( size_t row = 0; row < N_rows; row++ )
            for ( size_t col = 0; col < N_cols; col++ )
            {
                weights[feature](row,col) = x[edgeWeightsMap[feature](row,col)];
            }
    }

}


/*------------------------------------------------------------------------------

                              l2Regularization

------------------------------------------------------------------------------*/

double l2Regularization( const double *x, double *g , size_t n, CTrainingDataSet *td )
{
    TTrainingOptions &to = td->getTrainingOptions();

    // Apply L-2 norm
    double regularizationFactor = 0;
    for ( size_t i = 0; i < n; i++ )
    {
        regularizationFactor += to.lambda[i]*(x[i]*x[i]);
        g[i] += 2*to.lambda[i]*x[i];
    }

    return regularizationFactor;

}


/*------------------------------------------------------------------------------

                                evaluate

------------------------------------------------------------------------------*/

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{

    using namespace UPGMpp;
    CTrainingDataSet *td = static_cast<UPGMpp::CTrainingDataSet*>(instance);

    // Reset the vector of gradients
    for ( size_t i = 0; i < n; i++ )
        g[i] = 0;

    // Update the node and edge weights
    vector<CNodeTypePtr> &nodeTypes = td->getNodeTypes();
    vector<CEdgeTypePtr> &edgeTypes = td->getEdgeTypes();

    size_t N_nodeTypes = nodeTypes.size();
    size_t N_edgeTypes = edgeTypes.size();

    for ( size_t index = 0; index < N_nodeTypes; index++ )
        updateNodeTypeWeights( td->getCertainNodeWeightsMap( nodeTypes[index] ),
                           nodeTypes[index],
                           x);

    for ( size_t index = 0; index < N_edgeTypes; index++ )
        updateEdgeTypeWeights( td->getCertainEdgeWeightsMap( edgeTypes[index] ),
                           edgeTypes[index],
                           x);

    // For each graph in the dataset

   /* for ( size_t i=0; i < n; i++ )
        cout << "x[" << i << "] : " << x[i] << endl;*/

    lbfgsfloatval_t fx = 0.0;


    vector<CGraph> & graphs = td->getGraphs();
    std::vector<std::map<size_t,size_t> > &groundTruth = td->getGroundTruth();

    size_t N_datasets = graphs.size();

    // Compute the function value

    TTrainingOptions &to = td->getTrainingOptions();

    for ( size_t dataset = 0; dataset < N_datasets; dataset++ )
    {        
        graphs[dataset].computePotentials();

        if ( to.trainingType == "pseudolikelihood" )
            td->updatePseudolikelihood( graphs[dataset], groundTruth[dataset], fx, x, g );
        else if ( to.trainingType == "inference" )
            td->updateInference( graphs[dataset], groundTruth[dataset], fx, x, g );
        else if ( to.trainingType == "decoding" )
            td->updateDecoding( graphs[dataset], groundTruth[dataset], fx, x, g );
    }    

    //for ( size_t i=0; i < n; i++ )
    //       cout << "g[" << i << "] : " << g[i] << endl;

    //for ( size_t i=0; i < n; i++ )
    //       cout << "x[" << i << "] : " << x[i] << endl;

    //cout << "fx            : " << fx << endl;
    if ( to.l2Regularization )
        fx = fx + l2Regularization( x, g, n, td );

    //cout << "fx regularized: " << fx << endl;

    return fx;
}

/*------------------------------------------------------------------------------

                                progress

------------------------------------------------------------------------------*/

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    using namespace UPGMpp;
    CTrainingDataSet *td = static_cast<CTrainingDataSet*>(instance);
    TTrainingOptions &to = td->getTrainingOptions();

    if ( to.showTrainingProgress )
    {
        cout << "Iteration " << k << endl;
        cout << "  fx = " <<  fx << ", x[0] = " << x[0] << ", x[1] = " << x[1] << endl;
        cout << "  xnorm = " << xnorm << ", gnorm = " << gnorm << ", step = " << step << endl << endl;

    }

    return 0;
}


/*------------------------------------------------------------------------------

                                train

------------------------------------------------------------------------------*/

int CTrainingDataSet::train( const bool debug )
{
    // Steps of the train algorithm:
    //  1. Build the mapping between the different types of nodes and edges and
    //      the positions of the vector of weights to optimize.
    //  2. Initialize the weights.
    //  3. Configure the parameters of the optimization method.
    //  4. Launch optimization (training).
    //  5. Show optimization results.

    //
    //  1. Build the mapping between the different types of nodes and edges and
    //      the positions of the vector of weights to optimize.
    //

    // Nodes

    DEBUG("Preparing weights of the different node types...",1);

    N_weights = 0;
    for ( size_t i = 0; i < m_nodeTypes.size(); i++ )
    {
        size_t N_cols = m_nodeTypes[i]->getWeights().cols();
        size_t N_rows = m_nodeTypes[i]->getWeights().rows();

        Eigen::MatrixXi weightsMap;
        weightsMap.resize( N_rows, N_cols );

        size_t index = N_weights;

        for ( size_t row = 0; row < N_rows; row++)
            for ( size_t col = 0; col < N_cols; col++ )
            {
                weightsMap(row,col) = index;
                index++;
            }

        m_nodeWeightsMap[m_nodeTypes[i]] = weightsMap;

        N_weights += N_cols * N_rows;

        //cout << *m_nodeTypes[i] << endl;

    }

    // Edges

    DEBUG("Preparing weights of the different edge types...",1);

    for ( size_t i = 0; i < m_edgeTypes.size(); i++ )
    {
        size_t N_features = m_edgeTypes[i]->getWeights().size();

        std::vector<Eigen::MatrixXi> v_weightsMap(N_features);

        size_t index = N_weights;

        Eigen::VectorXi typeOfEdgeFeatures =
                                          m_typesOfEdgeFeatures[m_edgeTypes[i]];

        for ( size_t feature = 0; feature < N_features; feature++ )
        {
            if ( typeOfEdgeFeatures(feature) == 0 )
            {
                size_t N_cols = m_edgeTypes[i]->getWeights()[feature].cols();
                size_t N_rows = m_edgeTypes[i]->getWeights()[feature].rows();

                Eigen::MatrixXi weightsMap;
                weightsMap.resize( N_rows, N_cols );

                weightsMap(0,0) = index;

                for ( size_t row = 0; row < N_rows; row++ ) // rows
                {
                    index++;

                    for ( size_t col = row+1; col < N_cols; col++ )  // cols
                    {
                        weightsMap(row,col) = index;
                        weightsMap(col,row) = index;
                        index++;
                    }
                }

                size_t previousW = N_weights;

                for ( size_t c3 = 1; c3 < N_rows; c3++ )
                {
                    index = previousW + (N_rows - c3) + 1;

                    weightsMap(c3,c3) = index;
                    previousW = index;
                }

                index++;

                N_weights = weightsMap(N_rows-1,N_cols-1)+1;

                v_weightsMap[feature] = weightsMap;

            }
            else if ( typeOfEdgeFeatures(feature) == 1 )
            {
                size_t N_cols = m_edgeTypes[i]->getWeights()[feature].cols();
                size_t N_rows = m_edgeTypes[i]->getWeights()[feature].rows();


                Eigen::MatrixXi weightsMap;
                weightsMap.resize( N_rows, N_cols );

                for ( size_t row = 0; row < N_rows; row++)
                    for ( size_t col = 0; col < N_cols; col++ )
                    {
                        weightsMap(row,col) = index;
                        index++;
                    }

                v_weightsMap[feature] = weightsMap;

                N_weights += N_cols * N_rows;
            }
            else if ( typeOfEdgeFeatures(feature) == 2 )
            {
                v_weightsMap[feature] = v_weightsMap[feature-1].transpose();
            }

            //cout << "Map for feature" << feature << endl;
            //cout << v_weightsMap[feature] << endl;
        }

        m_edgeWeightsMap[m_edgeTypes[i]] = v_weightsMap;

        //cout << *m_edgeTypes[i] << endl;
    }

    //cout << "Number of weights " << N_weights << endl;
    DEBUGD("Number of weights",N_weights,2);

    //
    //  2. Initialize the weights.
    //

    // Initialize weights
    lbfgsfloatval_t *x = lbfgs_malloc(N_weights);

    for ( size_t i = 0; i < N_weights; i++ )
        x[i] = 0;

    //
    //  3. Configure the parameters of the optimization method.
    //

    lbfgsfloatval_t fx;
    lbfgs_parameter_t param;

    lbfgs_parameter_init(&param);

    if ( m_trainingOptions.l2Regularization )
    {
        m_trainingOptions.lambda.resize(N_weights);

        Eigen::MatrixXi &lastNodeMapMatrix = m_nodeWeightsMap[m_nodeTypes[m_nodeTypes.size()-1]];

        size_t lastNodeWeight =
                lastNodeMapMatrix(lastNodeMapMatrix.rows()-1,lastNodeMapMatrix.cols()-1);

        for ( size_t i = 0; i < N_weights; i++ )
            if ( i <= lastNodeWeight )
                m_trainingOptions.lambda[i] = m_trainingOptions.nodeLambda;
            else
                m_trainingOptions.lambda[i] = m_trainingOptions.edgeLambda;
    }


    if (x == NULL) {
        cout << "ERROR: Failed to allocate a memory block for variables." << endl;
    }    

    DEBUG("Initializing parameters...",1);

    /* Initialize the parameters for the L-BFGS optimization. */
    //param.orthantwise_c = 100;
    //param.orthantwise_start = 1;
    //param.orthantwise_end = N_weights - 1;

    int &lsm = m_trainingOptions.linearSearchMethod;
    if ( !lsm )
        param.linesearch = LBFGS_LINESEARCH_DEFAULT;
    else if ( lsm == 1 )
        param.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
    else if ( lsm == 2 )
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    else if ( lsm == 3 )
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    else if ( lsm == 4 )
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
    else if ( lsm == 5 )
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

    param.max_iterations = m_trainingOptions.maxIterations;

    param.xtol = 10*exp(-31);
    param.ftol = 10*exp(-31);
    param.gtol = 10*exp(-31);
    param.max_step = 10*exp(31);



    //
    //  4. Launch optimization (training).
    //

    DEBUG("Optimizing...",1);

    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */

    int ret = lbfgs(N_weights, x, &fx, evaluate, progress, this, &param);

    //
    //  5. Show optimization results.
    //

    if (ret == LBFGS_CONVERGENCE) {
        cout << "L-BFGS resulted in convergence" << endl;
    } else if (ret == LBFGS_STOP) {
        cout << "L-BFGS terminated with the stopping criteria" << endl;
    } else if (ret == LBFGSERR_MAXIMUMITERATION) {
        cout << "L-BFGS terminated with the maximum number of iterations" << endl;
    } else {
        cout << "L-BFGS terminated with error code : " << ret << endl;
    }

    cout << "  fx = " << fx << endl;

    if ( m_trainingOptions.showTrainedWeights )
    {
        cout << "----------------------------------------------" << endl;

        cout << "Final vector of weights (x): " << endl;
        for ( size_t i=0; i < N_weights; i++ )
                cout << "x[" << i << "] : " << x[i] << endl;

        cout << "----------------------------------------------" << endl;

        for ( size_t i = 0; i < m_edgeTypes.size(); i++ )
        {
            cout << "Edge type " << m_edgeTypes[i]->getID() << endl;
            for ( size_t feat = 0; feat < m_edgeTypes[i]->getWeights().size(); feat++)
                cout << "Feature " << feat << " weights: " << endl << m_edgeTypes[i]->getWeights()[feat] << endl;

            cout << "----------------------------------------------" << endl;
        }

        for ( size_t i = 0; i < m_nodeTypes.size(); i++ )
        {
            cout << "Node type " << m_nodeTypes[i]->getID() << endl;
            cout << " weights: " << endl << m_nodeTypes[i]->getWeights() << endl;

            cout << "----------------------------------------------" << endl;
        }
    }

    lbfgs_free(x);

    return ret;
}


/*------------------------------------------------------------------------------

                        updateFunctionValueAndGradients

------------------------------------------------------------------------------*/

void CTrainingDataSet::updatePseudolikelihood( CGraph &graph,
                                                        std::map<size_t,size_t> &groundTruth,
                                                        lbfgsfloatval_t &fx,
                                                        const lbfgsfloatval_t *x,
                                                        lbfgsfloatval_t *g )
{
     //cout << "[STATUS] Updating function value and grandients! Graph: " << graph.getID() << endl ;
    vector<CNodePtr> &nodes = graph.getNodes();

    vector<CNodePtr>::iterator itNodes;

    // Computet the probability of each class of each node while their neighbors
    // take a fixed value
    for ( itNodes = nodes.begin(); itNodes != nodes.end(); itNodes++ )
    {
        CNodePtr node = *itNodes;

        // Get direct access to some interesting members of the node
        CNodeTypePtr    nodeType    = node->getType();
        Eigen::VectorXd potentials  = node->getPotentials();
        const Eigen::VectorXd &features = node->getFeatures();        

        size_t ID = node->getID();

        // Multiply the node potentias with the potentials of its neighbors
        pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator > neighbors;

        neighbors = graph.getEdgesF().equal_range(ID);

        for ( multimap<size_t,CEdgePtr>::iterator it = neighbors.first; it != neighbors.second; it++ )
        {
            size_t neighborID;
            size_t ID1, ID2;

            CEdgePtr        edgePtr ((*it).second);
            Eigen::MatrixXd edgePotentials = edgePtr->getPotentials();

            edgePtr->getNodesID(ID1,ID2);


            //cout << "JARRRR" << endl;
            //cout << "potentials" << potentials <<  endl;
            //cout << "edge potentials" << edgePotentials << endl;
            //cout << "edge: " << endl << *edgePtr << endl;

            if ( ID1 == ID ) // The neighbor node indexes the columns            
                potentials = potentials.cwiseProduct(
                                edgePotentials.col( groundTruth[ID2] )
                                );


            else // The neighbor node indexes the rows            
                potentials = potentials.cwiseProduct(
                                edgePotentials.row( groundTruth[ID1] ).transpose()
                                );

        }

        // Update objective funciton value!!!
        fx = fx - std::log( potentials(groundTruth[ID]) ) + std::log( potentials.sum() );

        // Update gradient

        //cout << "****************************************" << endl;
        //cout << "Potentials    : " << potentials.transpose() << endl;
        //cout << "Potentials sum:" << potentials.sum() << endl;
        Eigen::VectorXd nodeBel = potentials * ( 1 / (double) potentials.sum() );
        //cout << "Node bel      : " << nodeBel.transpose() << endl;

        size_t N_classes = potentials.rows();
        size_t N_features = features.rows();

        //cout << "N_Classes" << N_classes << " N_Features: " << N_features << endl;

        // Update node weights gradient
        for ( size_t class_i = 0; class_i < N_classes; class_i++ )
        {
            for ( size_t feature = 0; feature < N_features; feature++ )
            {
                size_t index = m_nodeWeightsMap[nodeType](class_i,feature);
                //cout << "Class" << class_i << " Feature: " << feature << endl;

                if ( index > 0 ) // is the weight set to 0?
                {
                    double ok = 0;

                    if ( class_i == groundTruth[ID] )
                    {
                        if ( m_trainingOptions.classRelevance && m_classesRelevance[nodeType].count() )
                            ok = 1*m_classesRelevance[nodeType](class_i);
                        else
                            ok = 1;
                    }



                    //cout << "----------------------------------------------" << endl;

                    //cout << "Previous g[" << index << "]" << g[index] << endl;

                    g[index] = g[index] + features(feature)*(nodeBel(class_i) - ok);

                    //cout << "Gradient at g[" << index << "]: " << g[index] << endl;

                    //cout << "Feature va lue: " << features(feature) << endl;
                    //cout << "Node bel     : " << nodeBel(feature) << endl;
                    //cout << "Ok           : " << ok << endl;
                }

            }
        }

        // Update the gradients of its edges weights
        for ( multimap<size_t,CEdgePtr>::iterator it = neighbors.first; it != neighbors.second; it++ )
        {
            size_t ID1, ID2;

            CEdgePtr        edgePtr = (*it).second;
            CEdgeTypePtr    edgeTypePtr = edgePtr->getType();
            size_t          N_edgeFeatures = edgePtr->getFeatures().rows();

            edgePtr->getNodesID(ID1,ID2);

            size_t rowsIndex, colsIndex;


            //cout << "N_classes : " << N_classes << endl;
            //cout << "N_edgeFeatures : " << N_edgeFeatures << endl;

            for ( size_t class_i = 0; class_i < N_classes; class_i++ )
            {
                if ( ID1 == ID )
                {
                    rowsIndex = class_i;
                    colsIndex = groundTruth[ID2];
                }
                else
                {
                    rowsIndex = groundTruth[ID1];
                    colsIndex = class_i;
                }

                for ( size_t feature = 0; feature < N_edgeFeatures; feature++ )
                {
                    size_t index;

                    index = m_edgeWeightsMap[edgeTypePtr][feature](rowsIndex,colsIndex);

                    if ( index > 0 )
                    {
                        double ok = 0;
                        if ( class_i == groundTruth[ID] )
                            ok = 1;

                        //cout << "Index: " << index << " Class index: " << class_i << " nodeBel: " << nodeBel.transpose() << endl;

                        g[index] = g[index] +
                                edgePtr->getFeatures()[feature] *
                                ( nodeBel(class_i) - ok );
                    }


                    //cout << "----------------------------------------------" << endl;
                    //cout << "Row index " << rowsIndex << " cols index " << colsIndex  << endl;
                    //cout << "Gradient at " << index <<": " << g[index] << endl;

                }
            }
        }
    }

    //cout << "Fx: " << fx << endl;
    //cout << "[STATUS] Function value and gradients updated!" << endl;

}

/*------------------------------------------------------------------------------

                            updateInference

------------------------------------------------------------------------------*/

void CTrainingDataSet::updateInference( CGraph &graph,
                                        std::map<size_t,size_t> &groundTruth,
                                        double &fx,
                                        const double *x,
                                        double *g )
{

    map<size_t,VectorXd> nodeBeliefs;
    map<size_t,MatrixXd> edgeBeliefs;
    double logZ;

    if ( m_trainingOptions.inferenceMethod == "LBP" )
    {
        CLBPInference LBPinfer;
        LBPinfer.infer( graph, nodeBeliefs, edgeBeliefs, logZ );
    }
    else
    {
        cout << "Unknown inference method specified for training." << endl;
        return;
    }

        // Update objective funciton value!!!
        fx = fx - graph.getUnnormalizedLogLikelihood(groundTruth) + logZ;

        // Update gradient

        //cout << "****************************************" << endl;
        //cout << "Node bel      : " << nodeBel.transpose() << endl;

        vector<CNodePtr> &v_nodes = graph.getNodes();
        size_t           N_nodes  = v_nodes.size();

        // Update nodes weights gradient
        for ( size_t node = 0; node < N_nodes; node++ )
        {
            CNodePtr nodePtr = v_nodes[node];

            const Eigen::VectorXd &features = nodePtr->getFeatures();
            size_t ID = nodePtr->getID();
            CNodeTypePtr nodeType = nodePtr->getType();
            VectorXd nodeBel = nodeBeliefs[nodePtr->getID()] ;

            size_t N_features = features.rows();
            size_t N_classes = nodeType->getNumberOfClasses();

            for ( size_t class_i = 0; class_i < N_classes; class_i++ )
            for ( size_t feature = 0; feature < N_features; feature++ )
            {
                size_t index = m_nodeWeightsMap[nodeType](class_i,feature);
                //cout << "Class" << class_i << " Feature: " << feature << endl;

                if ( index > 0 ) // is the weight set to 0?
                {
                    double ok = 0;

                    if ( class_i == groundTruth[ID] )
                        ok = 1;



                    //cout << "----------------------------------------------" << endl;
                    //cout << "Previous g[" << index << "]" << g[index] << endl;

                    g[index] = g[index] + features(feature)*(nodeBel(class_i) - ok);

                    //cout << "Gradient at g[" << index << "]: " << g[index] << endl;

                    //cout << "Feature va lue: " << features(feature) << endl;
                    //cout << "Node bel     : " << nodeBel(feature) << endl;
                    //cout << "Ok           : " << ok << endl;
                }
            }
        }

        // Update the edge gradients

        vector<CEdgePtr> &v_edges = graph.getEdges();
        size_t N_edges = v_edges.size();

        for ( size_t edge_i = 0; edge_i < N_edges; edge_i++ )
        {
            CEdgePtr edgePtr = v_edges[edge_i];
            CEdgeTypePtr edgeTypePtr = edgePtr->getType();
            size_t N_features = edgeTypePtr->getNumberOfFeatures();
            size_t edgeID = edgePtr->getID();

            MatrixXd edgeBel = edgeBeliefs[edgeID];

            CNodePtr node1Ptr, node2Ptr;
            edgePtr->getNodes( node1Ptr, node2Ptr );

            size_t N_classes1 = node1Ptr->getType()->getNumberOfClasses();
            size_t N_classes2 = node2Ptr->getType()->getNumberOfClasses();

            size_t ID1 = node1Ptr->getID();
            size_t ID2 = node2Ptr->getID();

            for ( size_t state1 = 0; state1 < N_classes1; state1++ )
            for (size_t state2 = 0; state2 < N_classes2; state2++ )
            for ( size_t feature = 0; feature < N_features; feature++ )
            {
                size_t index;
                index = m_edgeWeightsMap[edgeTypePtr][feature](state1,state2);

                double ok = 0;

                if ( ( state1 == groundTruth[ID1]) && ( state2 == groundTruth[ID2] ) )
                    ok = 1;

                g[index] = g[index] +
                              edgePtr->getFeatures()[feature] *
                             ( edgeBel(state1,state2) - ok );
            }

        }


    //cout << "Fx: " << fx << endl;
    //cout << "[STATUS] Function value and gradients updated!" << endl;

}


/*------------------------------------------------------------------------------

                            updateInference

------------------------------------------------------------------------------*/

void CTrainingDataSet::updateDecoding( CGraph &graph,
                                        std::map<size_t,size_t> &groundTruth,
                                        double &fx,
                                        const double *x,
                                        double *g )
{

    map<size_t,size_t> MAPResults;

    if ( m_trainingOptions.decodingMethod == "AlphaExpansions" )
    {
        CDecodeAlphaExpansion decodeAlphaExpansion;
        decodeAlphaExpansion.decode(graph,MAPResults);

        /*std::map<size_t,size_t>::iterator it;

        for ( it = MAPResults.begin(); it != MAPResults.end(); it++ )
        {
            std::cout << " " << it->second;
        }
        cout << std::endl;*/
    }
    else
    {
        cout << "Unknown decoding method specified for training." << endl;
        return;
    }

    vector<CNodePtr> &v_nodes = graph.getNodes();
    size_t N_nodes = v_nodes.size();
    vector<CEdgePtr> &v_edges = graph.getEdges();
    size_t N_edges = v_edges.size();

    map<size_t,VectorXd> nodeBeliefs;
    map<size_t,MatrixXd> edgeBeliefs;
    double logZ;

    for ( size_t i_node = 0; i_node < N_nodes; i_node++ )
    {
        CNodePtr nodePtr = v_nodes[i_node];
        size_t ID = nodePtr->getID();
        size_t N_classes = nodePtr->getType()->getNumberOfClasses();

        VectorXd nodeBel(N_classes);
        nodeBel.setZero();
        //nodeBel.setConstant(0.05);
        nodeBel( MAPResults[ID] ) = 0.5;

        nodeBeliefs[ID] = nodeBel;

        //cout << "Node beliefs " << ID << endl << nodeBeliefs[ID] << endl;
    }

    for ( size_t i_edge=0; i_edge < N_edges; i_edge++)
    {
        CEdgePtr edgePtr = v_edges[i_edge];
        size_t edgeID = edgePtr->getID();

        CNodePtr node1Ptr, node2Ptr;
        edgePtr->getNodes( node1Ptr, node2Ptr );

        size_t N_classes1 = node1Ptr->getType()->getNumberOfClasses();
        size_t N_classes2 = node2Ptr->getType()->getNumberOfClasses();

        size_t ID1 = node1Ptr->getID();
        size_t ID2 = node2Ptr->getID();

        MatrixXd edgeBel(N_classes1, N_classes2);
        edgeBel.setZero();
        //edgeBel.setConstant(0.05);
        edgeBel( MAPResults[ID1], MAPResults[ID2] ) = 0.5;

        edgeBeliefs[edgeID] = edgeBel;

        //cout << "Edge beliefs " << edgeID << endl << edgeBeliefs[edgeID] << endl;
    }

    //cout << "JARRRRR" << endl;

    logZ = graph.getUnnormalizedLogLikelihood( MAPResults );

    //cout << "logZ:       " << logZ << endl;
    //cout << "Likelihood: " << graph.getUnnormalizedLogLikelihood(groundTruth) << endl;

    // Update objective funciton value!!!
    fx = fx - graph.getUnnormalizedLogLikelihood(groundTruth) + logZ;

    // Update gradient

    // Update nodes weights gradient
    for ( size_t node = 0; node < N_nodes; node++ )
    {
        CNodePtr nodePtr = v_nodes[node];

        const Eigen::VectorXd &features = nodePtr->getFeatures();
        size_t ID = nodePtr->getID();
        CNodeTypePtr nodeType = nodePtr->getType();
        VectorXd nodeBel = nodeBeliefs[nodePtr->getID()] ;

        size_t N_features = features.rows();
        size_t N_classes = nodeType->getNumberOfClasses();

        for ( size_t class_i = 0; class_i < N_classes; class_i++ )
            for ( size_t feature = 0; feature < N_features; feature++ )
            {
                size_t index = m_nodeWeightsMap[nodeType](class_i,feature);
                //cout << "Class" << class_i << " Feature: " << feature << endl;

                if ( index > 0 ) // is the weight set to 0?
                {
                    double ok = 0;

                    if ( class_i == groundTruth[ID] )
                        ok = 1;



                    //cout << "----------------------------------------------" << endl;
                    //cout << "Previous g[" << index << "]" << g[index] << endl;

                    g[index] = g[index] + features(feature)*(nodeBel(class_i) - ok);

                    //cout << "Gradient at g[" << index << "]: " << g[index] << endl;

                    //cout << "Feature value: " << features(feature) << endl;
                    //cout << "Node bel     : " << nodeBel(feature) << endl;
                    //cout << "Ok           : " << ok << endl;


                }
            }
    }

    // Update the edge gradients

    for ( size_t edge_i = 0; edge_i < N_edges; edge_i++ )
    {
        CEdgePtr edgePtr = v_edges[edge_i];
        CEdgeTypePtr edgeTypePtr = edgePtr->getType();
        size_t N_features = edgeTypePtr->getNumberOfFeatures();
        size_t edgeID = edgePtr->getID();

        MatrixXd edgeBel = edgeBeliefs[edgeID];

        CNodePtr node1Ptr, node2Ptr;
        edgePtr->getNodes( node1Ptr, node2Ptr );

        size_t N_classes1 = node1Ptr->getType()->getNumberOfClasses();
        size_t N_classes2 = node2Ptr->getType()->getNumberOfClasses();

        size_t ID1 = node1Ptr->getID();
        size_t ID2 = node2Ptr->getID();

        for ( size_t state1 = 0; state1 < N_classes1; state1++ )
            for (size_t state2 = 0; state2 < N_classes2; state2++ )
                for ( size_t feature = 0; feature < N_features; feature++ )
                {
                    size_t index;
                    index = m_edgeWeightsMap[edgeTypePtr][feature](state1,state2);

                    double ok = 0;

                    if ( ( state1 == groundTruth[ID1]) && ( state2 == groundTruth[ID2] ) )
                        ok = 1;

                    g[index] = g[index] +
                            edgePtr->getFeatures()[feature] *
                            ( edgeBel(state1,state2) - ok );
                }

    }



    //cout << "Fx: " << fx << endl;
    //cout << "[STATUS] Function value and gradients updated!" << endl;

}
