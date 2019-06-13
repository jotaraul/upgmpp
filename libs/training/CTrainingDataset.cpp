
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

#include "CTrainingDataset.hpp"
#include <lbfgs.h>
#include "inference_MAP.hpp"
#include "inference_marginal.hpp"
#include <omp.h>
#include <vector>

using namespace std;
using namespace UPGMpp;

double evaluationTime;
int iterCount;


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
        if ( to.optimizationMethod == "LBFGS" )
        {
            regularizationFactor += to.lambda[i]*(x[i]*x[i]);
            g[i] += 2*to.lambda[i]*x[i];
        }
        else // SGD
        {
            regularizationFactor += to.lambda[i]*(x[i]*x[i])/(td->getGraphs().size()/(double)to.sgd.evaluationsPerStep);
            g[i] += 2*to.lambda[i]*x[i]/(td->getGraphs().size()/(double)to.sgd.evaluationsPerStep);
        }
    }

    return regularizationFactor;

}

/*------------------------------------------------------------------------------

                              l1Regularization

------------------------------------------------------------------------------*/

double l1Regularization( const double *x, double *g , size_t n, CTrainingDataSet *td )
{
    // This is a regularization in the form:
    // weight = weight + (1/beta)*abs(weight)
    // so lambda = 1/beta
    // the gradient is updated by (1/beta)*(weight/abs(weight))

    TTrainingOptions &to = td->getTrainingOptions();

    // Apply L-1 norm
    double regularizationFactor = 0;
    for ( size_t i = 0; i < n; i++ )
    {
        if ( !x[i] )
            continue;

        if ( to.optimizationMethod == "LBFGS" )
        {
            regularizationFactor += to.lambda[i]*abs(x[i]);
            g[i] += to.lambda[i]*(x[i]/abs(x[i]));
        }
        else
        {
            regularizationFactor += to.lambda[i]*abs(x[i])/(td->getGraphs().size()/(double)to.sgd.evaluationsPerStep);
            g[i] += to.lambda[i]*(x[i]/abs(x[i]))/(td->getGraphs().size()/(double)to.sgd.evaluationsPerStep);
        }
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
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());

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

    size_t N_graphs = graphs.size();

    // Compute the function value

    TTrainingOptions &to = td->getTrainingOptions();

#ifdef UPGMPP_USING_OMPENMP
    omp_set_dynamic(0);
    omp_set_num_threads( to.numberOfThreads );
#endif

#pragma omp parallel for reduction(+:fx) if(to.parallelize)

    for ( size_t dataset = 0; dataset < N_graphs; dataset++ )
    {
        graphs[dataset].computePotentials();

        if ( to.trainingType == "pseudolikelihood" )
            td->updatePseudolikelihood( graphs[dataset], groundTruth[dataset], fx, x, g );
        else if ( to.trainingType == "scoreMatching")
            td->updateScoreMatching( graphs[dataset], groundTruth[dataset], fx, x, g );
        else if ( to.trainingType == "picewise")
            td->updatePicewise( graphs[dataset], groundTruth[dataset], fx, x, g );
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

    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration(time_end - time_start);
    evaluationTime += duration.total_nanoseconds();

    if ( to.logTraining && !(iterCount%to.iterationResolution) )
    {
        double unLike = 0;

        for ( size_t dataset = 0; dataset < N_graphs; dataset++ )
        {
            graphs[dataset].computePotentials();
            unLike += graphs[dataset].getUnnormalizedLogLikelihood(groundTruth[dataset]);
        }

        float gNorm = 0;
        float xNorm = 0;

        for ( size_t j = 0; j < n; j++ )
        {
            gNorm += g[j]*g[j];
            xNorm += x[j]*x[j];
        }

        gNorm = sqrt(gNorm);
        xNorm = sqrt(xNorm);

        if ( xNorm < 1 )
            xNorm = 1;

        TTrainingLogEntry logEntry;
        logEntry.unLikelihood = unLike;
        logEntry.gradientNorm = gNorm;
        logEntry.convergence = gNorm/xNorm;
        logEntry.ellapsedTime = evaluationTime;
        logEntry.consumedData = (iterCount+1)*N_graphs;

        td->getTrainingLog().entries.push_back(logEntry);

    }

//    if (!(iterCount%1000))
//    {
//        double unLikelihood = 0;
//        for ( size_t dataset = 0; dataset < N_graphs; dataset++ )
//        {
//            graphs[dataset].computePotentials();
//            unLikelihood += graphs[dataset].getUnnormalizedLogLikelihood(groundTruth[dataset]);
//        }

//        cout << "Iter = " << iterCount << " likelihood = "  << unLikelihood << endl;
//    }

    iterCount++;

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

                                  sgd

------------------------------------------------------------------------------*/

int sgd( int N_weights, lbfgsfloatval_t *x, void *instance, bool debug )
{
    using namespace UPGMpp;
    CTrainingDataSet *td = static_cast<UPGMpp::CTrainingDataSet*>(instance);

    TTrainingLog &tlog = td->getTrainingLog();
    TTrainingOptions &to = td->getTrainingOptions();
    int &maxIter = to.maxIterations;
    double &stepSize = to.sgd.stepSize;

    std::srand(time(NULL));

    // Initialize gradients
    lbfgsfloatval_t *g    = lbfgs_malloc(N_weights);
    lbfgsfloatval_t *xTop = lbfgs_malloc(N_weights);
    double unLikelihoodTop = std::numeric_limits<double>::lowest();
    vector<double> v_unLikelihoods;
    lbfgsfloatval_t *pastIncrements = lbfgs_malloc(N_weights);
    lbfgsfloatval_t *outerProductDiag;

    // meta-descent
    lbfgsfloatval_t *gains;
    lbfgsfloatval_t *pastGradients;
    lbfgsfloatval_t *pastGradientsBis;
    lbfgsfloatval_t *pastSteps;

    // Initialize used vectors

    for ( size_t i = 0; i < N_weights; i++ )
        g[i] = 0;

    for ( size_t i = 0; i < N_weights; i++ )
        pastIncrements[i] = 0;

    if ( to.sgd.updateMethod == "adaptative" )
    {
        outerProductDiag = lbfgs_malloc(N_weights);
        for ( size_t i = 0; i < N_weights; i++ )
            outerProductDiag[i] = 0;
    }

    if ( to.sgd.updateMethod == "meta-descent" )
    {
        gains           = lbfgs_malloc(N_weights);
        pastGradients   = lbfgs_malloc(N_weights);
        pastGradientsBis= lbfgs_malloc(N_weights);
        pastSteps       = lbfgs_malloc(N_weights);
        for ( size_t i = 0; i < N_weights; i++ )
        {
            gains[i] = 0;
            pastGradients[i] = 0;
            pastGradientsBis[i] = 0;
            pastSteps[i] = stepSize;
        }
    }

    size_t iter = 0;
    bool convergence = false;
    double unLikelihood = 0;
    float xNormSum = 0;
    float gNormSum = 0;

    while ( iter < maxIter && !convergence )
    {
        DEBUGD("Doing iteration ", iter);

        boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());

        // Resent gradients
        for ( size_t i = 0; i < N_weights; i++ )
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

        // Get shortcuts
        vector<CGraph> & graphs = td->getGraphs();
        std::vector<std::map<size_t,size_t> > &groundTruth = td->getGroundTruth();

        size_t N_graphs = graphs.size();

        lbfgsfloatval_t fx = 0.0;

#ifdef UPGMPP_USING_OMPENMP
        omp_set_dynamic(0);
        omp_set_num_threads( (to.sgd.evaluationsPerStep>8) ? 8 : to.sgd.evaluationsPerStep );
#endif
        #pragma omp parallel for if(to.parallelize) //num_threads(4)
        for ( size_t evaluation = 0; evaluation < to.sgd.evaluationsPerStep; evaluation++ )
        {
            int graph;

            if ( to.sgd.samplePicking == "random" )
                graph = std::rand() % N_graphs;
            else if ( to.sgd.samplePicking == "sequential" )
            {
                graph = iter % N_graphs;
                if (to.parallelize)
                {
#ifdef UPGMPP_USING_OMPENMP
                    graph *= omp_get_num_threads();
                    graph += omp_get_thread_num();
#endif
                }
            }
            else
                cerr << "  [ERROR] Unkown sample picking strategy in SGD: "
                     << to.sgd.samplePicking << endl;

            // Compute the function value

            boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());

            DEBUG("  Computing graph potentials");

            graphs[graph].computePotentials();

            DEBUG("  Updating gradients")

            if ( to.trainingType == "pseudolikelihood" )
                td->updatePseudolikelihood( graphs[graph], groundTruth[graph], fx, x, g );
            else if ( to.trainingType == "scoreMatching")
                td->updateScoreMatching( graphs[graph], groundTruth[graph], fx, x, g );
            else if ( to.trainingType == "picewise")
                td->updatePicewise( graphs[graph], groundTruth[graph], fx, x, g );
            else if ( to.trainingType == "inference" )
                td->updateInference( graphs[graph], groundTruth[graph], fx, x, g );
            else if ( to.trainingType == "decoding" )
                td->updateDecoding( graphs[graph], groundTruth[graph], fx, x, g );
        }

//        // Check valid gradients
//        if (g[0] == -std::numeric_limits<double>::max())
//        {
//            iter++;
//            iterCount++;
//            continue;
//        }

        // Apply regularization
        //
        if ( to.l2Regularization )
            fx = fx + l2Regularization( x, g, N_weights, td );

        boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());

        double scheduledStep;
        if ( to.sgd.updateMethod == "schedule" )
             scheduledStep = (1 - log(1+(exp(1)-1)*(iter/static_cast<double>(maxIter))));

        // Update parameters
        //

        DEBUG("  Updating parameters");

        for ( size_t j = 1; j < N_weights; j++ )
        {
            double inc = - stepSize*g[j];

            if ( to.sgd.updateMethod == "momentum" )
            {
                inc += to.sgd.momentum_alpha*pastIncrements[j];
            }
            else if ( to.sgd.updateMethod == "adaptative" )
            {
                outerProductDiag[j] += g[j]*g[j];

                if ( outerProductDiag[j] )
                    inc = inc / sqrt(outerProductDiag[j]);
            }
            else if ( to.sgd.updateMethod == "schedule" )
            {
//                if (j == 10 )
//                cout << "Before: " << inc <<
//                      " Mult:" << (1 - log(1+(exp(1)-1)*(iter/static_cast<double>(maxIter))));
                //inc *= (1 - iter/static_cast<double>(maxIter));

                inc *= scheduledStep;
//                if ( j == 10 )cout << " After: " << inc << endl;

            }
            else if ( to.sgd.updateMethod == "meta-descent" )
            {
                double hessianApprox = g[j]-pastGradientsBis[j]/2;
                //double hessianApprox = g[j]-pastGradients[j];
                gains[j] = to.sgd.metaDescent_lambda*gains[j] -
                            pastSteps[j]*(pastGradients[j]+to.sgd.metaDescent_lambda*hessianApprox*gains[j]);
                double aux = 1-to.sgd.metaDescent_mu*g[j]*gains[j];

                if ( aux < 0.5 )
                    aux = 0.5;

                double step = pastSteps[j]*aux;

                // Compute inc
                inc = - step*g[j];

//                if(j==1)
//                cout << "hessian " << hessianApprox << " gains " << gains[j]
//                     << " pastSteps " << pastSteps[j] << " step " << step << " mu " << mu
//                     << " aux " << aux << " g " << g[j] << " inc " << inc << endl;

                // Update vbles
                pastSteps[j] = step;
                pastGradientsBis[j] = pastGradients[j];
                pastGradients[j] = g[j];
            }
            else if ( to.sgd.updateMethod != "standard")
            {
                cout << "  [ERROR] Undefined updating method for the SGD optimization." << endl;
                return -1;
            }

            pastIncrements[j] = inc;
            x[j] = x[j] + inc;
        }

        // Check convergence
        //
        float xNorm = 0;
        float gNorm = 0;

        for ( size_t j = 0; j < N_weights; j++ )
        {
            xNorm += x[j]*x[j];
            gNorm += g[j]*g[j];
        }

        xNorm = sqrt(xNorm);
        gNorm = sqrt(gNorm);

        xNormSum += xNorm;
        gNormSum += gNorm;

        // Show trainig process
        //
        if ( !(iter%2000) )
        {
            //if ( to.showTrainingProgress  )
            {
                // Evaluate likelihood
                //
                double pastUnLikelihood = unLikelihood;
                unLikelihood = 0;

                for ( size_t dataset = 0; dataset < N_graphs; dataset++ )
                {
                    graphs[dataset].computePotentials();
                    unLikelihood += graphs[dataset].getUnnormalizedLogLikelihood(groundTruth[dataset]);
                }

                double diff = unLikelihood - pastUnLikelihood;

                cout << "Iter = " << iterCount << " likelihood = "  << unLikelihood <<
                        " diff = " <<  diff <<
                        " gNorm = " << gNormSum << " xNorm = " << xNorm <<
                        " gNorm/xNorm = " << gNormSum/xNormSum << " fx = " << fx << endl;
            }

            cout.flush();
        }

        if ( !(iterCount%(int)(to.sgd.storeProgressEach) ))
        {
            double unLike = 0;

            for ( size_t dataset = 0; dataset < N_graphs; dataset++ )
            {
                graphs[dataset].computePotentials();
                unLike += graphs[dataset].getUnnormalizedLogLikelihood(groundTruth[dataset]);
            }

            if (isinf(unLike))
                v_unLikelihoods.push_back(v_unLikelihoods[v_unLikelihoods.size()-1]);
            else
                v_unLikelihoods.push_back(unLike);            

            if (( unLikelihoodTop < unLike ) && !isinf(unLike))
            {
                unLikelihoodTop = unLike;
                for ( size_t i = 0; i < N_weights; i++ )
                    xTop[i] = x[i];
            }
        }

        if ( iterCount && (iterCount>to.sgd.checkConvergencyFrom))
        {
            size_t index = iterCount / to.sgd.storeProgressEach;
//            cout << "Iter count: " << iterCount << " index: " << index << " size:  " << v_unLikelihoods.size() << endl;
//            for ( size_t i =0; i < v_unLikelihoods.size(); i++ )
//                cout << v_unLikelihoods[i] << " " << endl;
//            cout << "Current: " << v_unLikelihoods.at(index) << endl;

            if ((v_unLikelihoods[index] < v_unLikelihoods[index-(int)(to.sgd.checkConvergencyEach/to.sgd.storeProgressEach)])
                     || isinf(v_unLikelihoods[index]) || isinf(gNorm) )
            {
                convergence = true;

                for ( size_t i = 0; i < N_weights; i++ )
                    x[i] = xTop[i];

                if (v_unLikelihoods[index] < v_unLikelihoods[index-(int)(to.sgd.checkConvergencyEach/to.sgd.storeProgressEach)])
                    cout << "Convergence achieved!!! likelihood " << v_unLikelihoods[index] << " replaced by " << unLikelihoodTop << endl;
                else if (isinf(v_unLikelihoods[index]))
                    cout << "Infinite value of likelihood found!!! likelihood " << v_unLikelihoods[index] << " replaced by " << unLikelihoodTop << endl;
                else if (isinf(gNorm))
                    cout << "Infinite value of gNorm found!!! likelihood " << v_unLikelihoods[index] << " replaced by " << unLikelihoodTop << endl;
            }
        }

        boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
        boost::posix_time::time_duration duration(time_end - time_1);
        evaluationTime += duration.total_nanoseconds();

        //cout << "Time 1 " << boost::posix_time::time_duration(time_2 - time_1).total_nanoseconds() << endl;
        // cout << "Time 2 " << boost::posix_time::time_duration(time_3 - time_2).total_nanoseconds() << endl;
        //cout << "Time 3 " << boost::posix_time::time_duration(time_end - time_3).total_nanoseconds() << endl;

        if ( to.logTraining && !(iterCount%to.iterationResolution) )
        {
            double unLike = 0;

            for ( size_t dataset = 0; dataset < N_graphs; dataset++ )
            {
                graphs[dataset].computePotentials();
                unLike += graphs[dataset].getUnnormalizedLogLikelihood(groundTruth[dataset]);
            }

            if (!isinf(unLike) && !isnan(unLike))
            {

                TTrainingLogEntry logEntry;
                logEntry.unLikelihood = unLike;
                logEntry.gradientNorm = gNormSum;
                logEntry.convergence = gNormSum/xNormSum;
                logEntry.ellapsedTime = evaluationTime;
                logEntry.consumedData = (iter+1)*to.sgd.evaluationsPerStep;

                tlog.entries.push_back(logEntry);
            }
        }

        if ( !(iter%to.iterationResolution) )
        {
            double epsilon = 1e-6;

            if ( xNormSum < 1 )
                xNormSum = 1;

//            if ( gNormSum/xNormSum < epsilon )
//            {
//                convergence = true;
//                cout << "CONVERGEDDDDDDDDDDDDDDDDDDDDDDDDDDDD! gNorm: " << gNorm << " xNorm: " << xNorm << " Div: " << gNorm/xNorm << endl;
//            }

            xNormSum = 0;
            gNormSum = 0;
        }

        iter++;
        iterCount++;
    }

    // Release memory
    //
    lbfgs_free(g);
    lbfgs_free(pastIncrements);
    if ( to.sgd.updateMethod == "adaptative" ) lbfgs_free(outerProductDiag);
    if ( to.sgd.updateMethod == "meta-descent" )
    {
        lbfgs_free(gains);
        lbfgs_free(pastGradients);
        lbfgs_free(pastGradientsBis);
        lbfgs_free(pastSteps);
    }
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

    TIMER_START

    DEBUG("Preparing weights of the different node types...")

    N_weights = 1;
    for ( size_t i = 0; i < m_nodeTypes.size(); i++ )
    {
        size_t N_cols = m_nodeTypes[i]->getWeights().cols();
        size_t N_rows = m_nodeTypes[i]->getWeights().rows();

        if ( N_cols == 0 )
            // No features in this node type, probably due to handcoded potentials
            continue;


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

    DEBUG("Preparing weights of the different edge types...");

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
                // Check that a previous feature exists
                assert(feature!=0);

                v_weightsMap[feature] = v_weightsMap[feature-1].transpose();
            }
            else if ( typeOfEdgeFeatures(feature) == 3 ) // Weights only in the diagonal
            {
                size_t N_cols = m_edgeTypes[i]->getWeights()[feature].cols();
                size_t N_rows = m_edgeTypes[i]->getWeights()[feature].rows();

                if ( N_cols != N_rows )
                    std::logic_error("Non square matrix assigned type of weights 3");

                Eigen::MatrixXi weightsMap;
                weightsMap.resize( N_rows, N_rows );
                weightsMap.fill(std::numeric_limits<float>::min());

                for ( size_t row = 0; row < N_rows; row++)
                {
                    weightsMap(row,row) = index;
                    index++;
                }

                v_weightsMap[feature] = weightsMap;

                N_weights += N_rows;
            }

//            cout << "Map for feature" << feature << endl;
//            cout << v_weightsMap[feature] << endl;
        }

        m_edgeWeightsMap[m_edgeTypes[i]] = v_weightsMap;

        //cout << *m_edgeTypes[i] << endl;
    }

    //cout << "Number of weights " << N_weights << endl;
    DEBUGD("Number of weights ",N_weights);

    //
    //  2. Initialize the weights.
    //

    // Initialize weights
    lbfgsfloatval_t *x;
    if ( !m_x )
    {
        m_x = lbfgs_malloc(N_weights);
        x = (lbfgsfloatval_t*)m_x;
        for ( size_t i = 0; i < N_weights; i++ )
            x[i] = 0;
    }
    else if ( !m_trainingOptions.continueTraining )
    {
        x = m_x;
        for (size_t i = 0; i < N_weights; i++)
            x[i] = 0;
    }
    else
    {
        x = m_x;
    }

    //
    //  3. Configure the parameters of the optimization method.
    //

    m_trainingLog.iterationResolution = m_trainingOptions.iterationResolution;

    lbfgsfloatval_t fx;
    lbfgs_parameter_t param;

    lbfgs_parameter_init(&param);

    if ( m_trainingOptions.l2Regularization )
    {
        m_trainingOptions.lambda.resize(N_weights);
        int lastNodeWeight;

        if ( m_nodeWeightsMap.empty() )
        {
            lastNodeWeight = -1;
            DEBUG("No node weights, so nothing to regularize for them!")
        }
        else
        {

            Eigen::MatrixXi &lastNodeMapMatrix =
                    m_nodeWeightsMap[m_nodeTypes[m_nodeTypes.size()-1]];

            lastNodeWeight =
                    lastNodeMapMatrix(lastNodeMapMatrix.rows()-1,lastNodeMapMatrix.cols()-1);
        }

        for ( size_t i = 0; i < N_weights; i++ )
            if ( i <= lastNodeWeight )
                m_trainingOptions.lambda[i] = m_trainingOptions.nodeLambda;
            else
                m_trainingOptions.lambda[i] = m_trainingOptions.edgeLambda;
    }


    if (x == NULL) {
        cout << "ERROR: Failed to allocate a memory block for variables." << endl;
    }

    evaluationTime = 0;
    iterCount      = 0;

    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */

    int ret;

    if ( m_trainingOptions.optimizationMethod == "LBFGS" ||
         m_trainingOptions.optimizationMethod == "hybrid" )
    {
        DEBUG("Performing LBFGS optimization...");

        if (m_trainingOptions.optimizationMethod == "hybrid")
            sgd(N_weights, x, this, debug );

        DEBUG("Initializing parameters...");

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
        param.max_linesearch = 100;

        //param.xtol = 10*exp(-31);
        //param.ftol = 10*exp(-31);
        param.gtol = 0.1;
        //param.max_step = 10*exp(31);

        //
        //  4. Launch optimization (training).
        //

        DEBUG("Optimizing...");

        ret = lbfgs(N_weights, x, &fx, evaluate, progress, this, &param);

        DEBUG("Completed! showing results...");

        //
        //  5. Show optimization results.
        //

        if (ret == LBFGS_CONVERGENCE) {
            cout << "L-BFGS resulted in convergence" << endl;
        } else if (ret == LBFGS_STOP) {
            cout << "L-BFGS terminated with the stopping criteria" << endl;
        } else if (ret == LBFGSERR_MAXIMUMITERATION) {
            cout << "L-BFGS terminated with the maximum number of iterations" << endl;
        } else if (ret == LBFGSERR_MAXIMUMLINESEARCH ) {
            cout << "The line-search routine reaches the maximum number of evaluations" << endl;
        } else if (ret == LBFGSERR_MINIMUMSTEP ) {
            cout << "The line-search step became smaller than lbfgs_parameter_t::min_step" << endl;
        } else if (ret == LBFGSERR_ROUNDING_ERROR ) {
            cout << "A rounding error occurred; alternatively, no line-search step "
                    "satisfies the sufficient decrease and curvature conditions" << endl;
        } else {
            cout << "L-BFGS terminated with error code : " << ret << endl;
        }

        cout << "  fx = " << fx << endl;
    }
    else if ( m_trainingOptions.optimizationMethod == "SGD" )
    {
        DEBUG("Performing SGD optimization...");

        sgd(N_weights, x, this, debug );
    }

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

    if (!m_trainingOptions.continueTraining)
    {
        lbfgs_free(m_x);
        m_x = NULL;
    }

    TIMER_END(m_executionTime)

            m_evaluatingTime = evaluationTime;

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
    //cout << "[STATUS] Updating function value and grandients! Graph: " << graph.getID() << endl;
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
        //fx = fx - pow(1-std::log( potentials(groundTruth[ID]) ) + std::log( potentials.sum() ),2); // Ration matching

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

                    double value = features(feature)*(nodeBel(class_i) - ok);

#pragma omp atomic
                    g[index] += value;


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

                        double value = edgePtr->getFeatures()[feature] *
                                ( nodeBel(class_i) - ok );

#pragma omp atomic
                        g[index] += value;

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

                            updateScoreMatching

------------------------------------------------------------------------------*/

void CTrainingDataSet::updateScoreMatching( CGraph &graph,
                                            std::map<size_t,size_t> &groundTruth,
                                            lbfgsfloatval_t &fx,
                                            const lbfgsfloatval_t *x,
                                            lbfgsfloatval_t *g )
{
    //cout << "[STATUS] Updating function value and grandients! Graph: " << graph.getID() << endl;
    vector<CNodePtr> &nodes = graph.getNodes();

    vector<CNodePtr>::iterator itNodes;

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

            if ( ID1 == ID ) // The neighbor node indexes the columns
                potentials = potentials.cwiseProduct(
                            edgePotentials.col( groundTruth[ID2] )
                            );


            else // The neighbor node indexes the rows
                potentials = potentials.cwiseProduct(
                            edgePotentials.row( groundTruth[ID1] ).transpose()
                            );
        }

        double Z = potentials.sum();
        VectorXd nodeBel = potentials/Z;

        size_t N_classes = potentials.rows();
        size_t N_features = features.rows();

        // Update objective funciton value!!!
        for ( size_t class_i = 0; class_i < N_classes; class_i++ )
        {
            double ok = 0;
            if ( class_i == groundTruth[ID] )
                ok = 1;

            fx += pow(nodeBel(class_i)-ok,2);
        }

        // Update gradient

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
                        ok = 1;

                    for ( size_t class_j = 0; class_j < N_classes; class_j++ )
                    {
                        double value;
                        double ok2 = 0;
                        if ( class_j == groundTruth[ID] )
                            ok2 = 1;

                        if ( class_j == class_i )
                            value = 2*(nodeBel(class_i)-ok)*features(feature)*(potentials(class_i)/Z-pow(potentials(class_i),2)/pow(Z,2));
                        else
                            value = -2*(nodeBel(class_j)-ok2)*features(feature)*potentials(class_i)*(potentials(class_j)/pow(Z,2));
 #pragma omp atomic
                        g[index] += value;
                    }
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

                        for ( size_t class_j = 0; class_j < N_classes; class_j++ )
                        {
                            double ok2 = 0;
                            double value;

                            if ( class_j == groundTruth[ID] )
                                ok2 = 1;

                            if ( class_j == class_i )
                                value = 2*(nodeBel(class_i)-ok)*edgePtr->getFeatures()[feature]*(potentials(class_i)/Z-pow(potentials(class_i),2)/pow(Z,2));
                            else
                                value = -2*(nodeBel(class_j)-ok2)*edgePtr->getFeatures()[feature]*potentials(class_i)*potentials(class_j)/pow(Z,2);

#pragma omp atomic
                            g[index] += value;
                        }
                    }
                }
            }
        }
    }

    //cout << "Fx: " << fx << endl;
    //cout << "[STATUS] Function value and gradients updated!" << endl;

}


/*------------------------------------------------------------------------------

                             updatePicewise

------------------------------------------------------------------------------*/

void CTrainingDataSet::updatePicewise( CGraph &graph,
                                       std::map<size_t,size_t> &groundTruth,
                                       lbfgsfloatval_t &fx,
                                       const lbfgsfloatval_t *x,
                                       lbfgsfloatval_t *g )
{
    //cout << "[STATUS] Updating function value and grandients! Graph: " << graph.getID() << endl;

    vector<CEdgePtr> &edges = graph.getEdges();

    vector<CEdgePtr>::iterator itEdges;

    //cout << "N_edges: " << edges.size() << endl;

    // Iterate over all the edges
    for ( itEdges = edges.begin(); itEdges != edges.end(); itEdges++ )
    {
        CEdgePtr edge = *itEdges;

        CNodePtr n1,n2;
        size_t ID1, ID2;

        edge->getNodesID(ID1,ID2);
        edge->getNodes(n1,n2);

        size_t N_classes1 = n1->getPotentials().rows();
        size_t N_classes2 = n2->getPotentials().rows();

        // Compute potentials

        MatrixXd potentials(N_classes1,N_classes2);

        for ( size_t class_i = 0; class_i < N_classes1; class_i++)
            for ( size_t class_j = 0; class_j < N_classes2; class_j++)
                potentials(class_i,class_j)=n1->getPotentials()(class_i)*n2->getPotentials()(class_j)*edge->getPotentials()(class_i,class_j);

        double Z = potentials.sum();

        // Update function
        fx = fx - log(potentials(groundTruth[ID1],groundTruth[ID2])) + log(Z);

        //
        // Update gradients
        //

        // First node

        VectorXd nodeBel = potentials.rowwise().sum()/Z;

        VectorXd &feat = n1->getFeatures();

        for ( size_t class_i = 0; class_i < N_classes1; class_i++)
        {
            for ( size_t feat_i = 0; feat_i < feat.rows(); feat_i++ )
            {
                size_t index = m_nodeWeightsMap[n1->getType()](class_i,feat_i);

                if ( index > 0 )
                {

                    double ok = 0;
                    if ( class_i == groundTruth[ID1] )
                        ok = 1;

                    double value = feat[feat_i]*(nodeBel(class_i)-ok);

#pragma omp atomic
                    g[index] += value;
                }
            }
        }

        // Second node

        nodeBel = potentials.colwise().sum()/Z;

        VectorXd &feat2 = n2->getFeatures();

        for ( size_t class_j = 0; class_j < N_classes2; class_j++)
        {
            for ( size_t feat_i = 0; feat_i < feat2.rows(); feat_i++ )
            {
                size_t index = m_nodeWeightsMap[n2->getType()](class_j,feat_i);

                if ( index > 0 )
                {

                    double ok = 0;
                    if ( class_j == groundTruth[ID2] )
                        ok = 1;

                    double value = feat2[feat_i]*(nodeBel(class_j)-ok);

#pragma omp atomic
                    g[index] += value;
                }
            }
        }

        // Edge

        MatrixXd edgeBel = potentials / Z;
        double N_edgeFeatures = edge->getFeatures().rows();

        for ( size_t class_i = 0; class_i < N_classes1; class_i++)
            for ( size_t class_j = 0; class_j < N_classes2; class_j++)
                for ( size_t feature = 0; feature < N_edgeFeatures; feature++ )
                {
                    size_t index;

                    index = m_edgeWeightsMap[edge->getType()][feature](class_i,class_j);

                    if ( index > 0 )
                    {
                        double ok = 0;
                        if ( class_i == groundTruth[ID1] && class_j == groundTruth[ID2])
                            ok = 1;


                        double value = edge->getFeatures()[feature]*(edgeBel(class_i,class_j)-ok);

#pragma omp atomic
                            g[index] += value;
                        }
                    }
                }

    //cout << "Fx: " << fx << endl;


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
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    map<size_t,VectorXd> nodeBeliefs;
    map<size_t,MatrixXd> edgeBeliefs;
    double logZ;

    if ( m_trainingOptions.inferenceMethod == "LBP" )
    {
        CLBPInferenceMarginal LBPinfer;
        LBPinfer.infer( graph, nodeBeliefs, edgeBeliefs, logZ );
    }
    else if ( m_trainingOptions.inferenceMethod == "TRPBP" )
    {
        CTRPBPInferenceMarginal TRPBPinfer;
        TRPBPinfer.infer( graph, nodeBeliefs, edgeBeliefs, logZ );
    }
    else if ( m_trainingOptions.inferenceMethod == "RBP" )
    {
        CRBPInferenceMarginal RBPinfer;
        RBPinfer.infer( graph, nodeBeliefs, edgeBeliefs, logZ );
    }
    else
    {
        cout << "Unknown inference method specified for training." << endl;
        return;
    }

//            cout << "pre fx: " << fx << endl;
//            cout << "logZ  : " << logZ << endl;
//            cout << "likelihood: " << graph.getUnnormalizedLogLikelihood(groundTruth) << endl;

            if ( isnan(logZ ))
            {
                g[0] = -std::numeric_limits<double>::max();
                return;
            }

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());

    // Update objective funciton value!!!
    fx = fx - graph.getUnnormalizedLogLikelihood(groundTruth,false) + logZ;

//            cout << "post fx: " << fx << endl;

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
//                    cout << "Previous g[" << index << "]" << g[index] << endl;
                    double value = features(feature)*(nodeBel(class_i) - ok);

#pragma omp atomic
                    g[index] += value;

//                    cout << "Gradient at g[" << index << "]: " << g[index] << endl;

//                    cout << "Feature value: " << features(feature) << endl;
//                    cout << "Node bel     : " << nodeBel(feature) << endl;
//                    cout << "Ok           : " << ok << endl;
                }
            }
    }

    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());

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

        for ( size_t feature = 0; feature < N_features; feature++ )
            for ( size_t state1 = 0; state1 < N_classes1; state1++ )
            for (size_t state2 = 0; state2 < N_classes2; state2++ )
                {
                    size_t index;
                    index = m_edgeWeightsMap[edgeTypePtr][feature](state1,state2);

                    double ok = 0;

                    if ( ( state1 == groundTruth[ID1]) && ( state2 == groundTruth[ID2] ) )
                        ok = 1;

                    double value = edgePtr->getFeatures()[feature]*(edgeBel(state1,state2) - ok);

//                    cout << "Previous g[" << index << "]" << g[index] << endl;
#pragma omp atomic                    
                    g[index] += value;

//                    cout << "Gradient at g[" << index << "]: " << g[index] << endl;

//                    cout << "Feature value: " << edgePtr->getFeatures()[feature] << endl;
//                    cout << "Edge bel     : " << edgeBel(state1,state2) << endl;
//                    cout << "Ok           : " << ok << endl;
                }
    }

    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());

//    cout << "Time 21 " << boost::posix_time::time_duration(time_1 - time_0).total_nanoseconds() << endl;
//    cout << "Time 22 " << boost::posix_time::time_duration(time_2 - time_1).total_nanoseconds() << endl;
//    cout << "Time 23 " << boost::posix_time::time_duration(time_3 - time_2).total_nanoseconds() << endl;

    //cout << "Fx: " << fx << endl;
    //cout << "[STATUS] Function value and gradients updated!" << endl;

}


/*------------------------------------------------------------------------------

                            updaterDecoding

------------------------------------------------------------------------------*/

void CTrainingDataSet::updateDecoding( CGraph &graph,
                                       std::map<size_t,size_t> &groundTruth,
                                       double &fx,
                                       const double *x,
                                       double *g )
{
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    map<size_t,size_t> MAPResults;
    static size_t times = 0;

    TInferenceOptions options;

    options.particularD["numberOfRestarts"] = 100;

    if ( times < m_trainingOptions.numOfRandomStarts )
    {
        options.initialAssignation = "Random";
        times++;
    }

    if ( m_trainingOptions.decodingMethod == "maxNodePot" )
    {
        CMaxNodePotInferenceMAP decodeMaxNodePot;
        decodeMaxNodePot.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "ICM" )
    {
        CICMInferenceMAP decodeICM;
        decodeICM.setOptions(options);
        decodeICM.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "WithRestarts" )
    {
        CRestartsInferenceMAP decodeRestarts;
        decodeRestarts.setOptions(options);
        decodeRestarts.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "ICMGreedy" )
    {
        CICMGreedyInferenceMAP decodeICMGreedy;
        decodeICMGreedy.setOptions(options);
        decodeICMGreedy.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "exactInference" )
    {
        CExactInferenceMAP decodeExact;
        decodeExact.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "LBP" )
    {
        CLBPInferenceMAP decodeLBP;
        decodeLBP.setOptions(options);
        decodeLBP.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "TRPBP" )
    {
        CTRPBPInferenceMAP decodeTRPBP;
        decodeTRPBP.setOptions(options);
        decodeTRPBP.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "RBP" )
    {
        CRBPInferenceMAP decodeRBP;
        decodeRBP.setOptions(options);
        decodeRBP.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "AlphaBetaSwap")
    {
        CAlphaBetaSwapInferenceMAP decodeAlphaBetaSwap;
        decodeAlphaBetaSwap.setOptions(options);
        decodeAlphaBetaSwap.infer(graph,MAPResults);
    }
    else if ( m_trainingOptions.decodingMethod == "AlphaExpansions" )
    {
        CAlphaExpansionInferenceMAP decodeAlphaExpansion;
        decodeAlphaExpansion.setOptions(options);
        decodeAlphaExpansion.infer(graph,MAPResults);

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

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());

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
        nodeBel( MAPResults[ID] ) = 1;

//        nodeBel.setConstant(0.1/(N_classes-1));
//        nodeBel( MAPResults[ID] ) = 0.9;

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
        edgeBel( MAPResults[ID1], MAPResults[ID2] ) = 1;

//        edgeBel.setConstant(0.1/(N_classes1*N_classes2-1));
//        edgeBel( MAPResults[ID1], MAPResults[ID2] ) = 0.9;

        edgeBeliefs[edgeID] = edgeBel;

        //cout << "Edge beliefs " << edgeID << endl << edgeBeliefs[edgeID] << endl;
    }

    logZ = graph.getUnnormalizedLogLikelihood( MAPResults );

    //cout << "logZ:       " << logZ << endl;
    //cout << "Likelihood: " << graph.getUnnormalizedLogLikelihood(groundTruth) << endl;

    // Update objective funciton value!!!
    fx = fx - graph.getUnnormalizedLogLikelihood(groundTruth) + logZ;

    // Update gradient

    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());

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
                    double value = features(feature)*(nodeBel(class_i) - ok);

#pragma omp atomic
                    g[index] += value;

                    //cout << "Gradient at g[" << index << "]: " << g[index] << endl;

                    //cout << "Feature value: " << features(feature) << endl;
                    //cout << "Node bel     : " << nodeBel(feature) << endl;
                    //cout << "Ok           : " << ok << endl;
                }
            }
    }

    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());

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

//#ifdef UPGMPP_USING_OMPENMP
//        omp_set_dynamic(0);
//        omp_set_num_threads(4);
//#endif
//#pragma omp parallel for //num_threads(4)
        for ( size_t state1 = 0; state1 < N_classes1; state1++ )
            for (size_t state2 = 0; state2 < N_classes2; state2++ )
                for ( size_t feature = 0; feature < N_features; feature++ )
                {
                    size_t index;
                    index = m_edgeWeightsMap[edgeTypePtr][feature](state1,state2);

                    double ok = 0;

                    if ( ( state1 == groundTruth[ID1]) && ( state2 == groundTruth[ID2] ) )
                        ok = 1;

                    double value = edgePtr->getFeatures()[feature] *
                            ( edgeBel(state1,state2) - ok );
#pragma omp atomic
                    g[index] += value;
                }

    }


    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());

    //cout << "Fx: " << fx << endl;
    //cout << "[STATUS] Function value and gradients updated!" << endl;

    //cout << "Time 21 " << boost::posix_time::time_duration(time_1 - time_0).total_nanoseconds() << endl;
    //cout << "Time 22 " << boost::posix_time::time_duration(time_2 - time_1).total_nanoseconds() << endl;
    //cout << "Time 23 " << boost::posix_time::time_duration(time_3 - time_2).total_nanoseconds() << endl;
    //cout << "Time 24 " << boost::posix_time::time_duration(time_4 - time_3).total_nanoseconds() << endl;

}
