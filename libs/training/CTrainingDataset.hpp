
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


#ifndef _UPGMpp_TRAINING_DATASET_
#define _UPGMpp_TRAINING_DATASET_

#include "base.hpp"

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

#include <iostream>
#include <boost/shared_ptr.hpp>

namespace UPGMpp
{

/*----------------------------------------------------------------------------
 *
 *                            CTrainingDataSet
 *
 *---------------------------------------------------------------------------*/

    struct TSGDOptions
    {
        double          stepSize;
        double          gradientDiffToConverge;
        int             evaluationsPerStep;
        std::string     updateMethod;
        double          storeProgressEach;
        double          checkConvergencyFrom;
        double          checkConvergencyEach;
        double          momentum_alpha;
        double          metaDescent_mu;
        double          metaDescent_lambda;
        std::string     samplePicking; // "random", "sequential"

        TSGDOptions() : evaluationsPerStep(1),
                        stepSize(1e-4),
                        gradientDiffToConverge(1e-12),
                        updateMethod("standard"),
                        storeProgressEach(10),
                        checkConvergencyFrom(1000),
                        checkConvergencyEach(1000),
                        momentum_alpha(0.1),
                        metaDescent_mu(0.1),
                        metaDescent_lambda(0.8),
                        samplePicking("random")

        {}
    };

    struct TTrainingOptions
    {
        bool            showTrainingProgress;
        bool            showTrainedWeights;
        bool            l2Regularization;
        double          nodeLambda;
        double          edgeLambda;
        int             linearSearchMethod;
        int             maxIterations;
        bool            classRelevance;
        int             numOfRandomStarts; // Used with Viterbi (MAP) training
        bool            parallelize; // Parallelize training? OMP must has been enabled
        bool            logTraining;
        int             iterationResolution;
        int             numberOfThreads; // Used in LBFGS
        TSGDOptions     sgd;
        std::string     trainingType;
        std::string     inferenceMethod;
        std::string     decodingMethod;
        std::string     optimizationMethod;
        std::vector<double>  lambda;        

        TTrainingOptions(): showTrainingProgress(true),
                            showTrainedWeights(false),
                            l2Regularization(false),
                            nodeLambda(0),
                            edgeLambda(0),
                            linearSearchMethod(0),
                            maxIterations(2000),
                            parallelize(true),                            
                            classRelevance(false),
                            trainingType("pseudolikelihood"),
                            inferenceMethod("LBP"),
                            decodingMethod("AlphaExpansions"),
                            optimizationMethod("LBFGS"),
                            numOfRandomStarts(0),                            
                            logTraining(false),
                            iterationResolution(1000),
                            numberOfThreads(1)
        {}
    };

    struct TTrainingLogEntry
    {
        double unLikelihood;
        double gradientNorm;
        double convergence; // gradientNorm / xNorm
        double ellapsedTime;
        int consumedData;
    };

    struct TTrainingLog
    {
        int iterationResolution;
        vector<TTrainingLogEntry> entries;
    };

    class CTrainingDataSet
    {
    private:
        std::vector<CGraph>                     m_graphs;
        std::vector<std::map<size_t,size_t> >   m_groundTruth;
        std::vector<CNodeTypePtr>               m_nodeTypes;
        std::vector<CEdgeTypePtr>               m_edgeTypes;
        std::map<CEdgeTypePtr,Eigen::VectorXi>  m_typesOfEdgeFeatures;
        std::map<CNodeTypePtr,Eigen::VectorXd>  m_classesRelevance;
        size_t                                  N_weights;
        std::map<CNodeTypePtr, Eigen::MatrixXi> m_nodeWeightsMap;
        std::map<CEdgeTypePtr,std::vector<Eigen::MatrixXi> > m_edgeWeightsMap;
        TTrainingOptions                        m_trainingOptions;
        TTrainingLog                            m_trainingLog;
        double                                  m_executionTime;
        double                                  m_evaluatingTime;

    public:

        CTrainingDataSet() : m_evaluatingTime(0), m_executionTime(0){}

        inline void setGroundTruth( std::vector<std::map<size_t,size_t> > &gt ) { m_groundTruth = gt; }
        inline void addGraphGroundTruth( std::map<size_t,size_t> &graph_gt ){ m_groundTruth.push_back( graph_gt ); }

        inline void setTrainingOptions( const TTrainingOptions &trainingOptions)
                                        { m_trainingOptions = trainingOptions; }
        inline TTrainingOptions& getTrainingOptions()
                                        { return m_trainingOptions; }
        inline TTrainingLog& getTrainingLog()
                                        { return m_trainingLog; }

        inline std::vector<CGraph>& getGraphs(){ return m_graphs; }
        inline std::vector<CNodeTypePtr>& getNodeTypes(){ return m_nodeTypes; }
        inline std::vector<CEdgeTypePtr>& getEdgeTypes(){ return m_edgeTypes; }
        inline Eigen::MatrixXi& getCertainNodeWeightsMap( CNodeTypePtr nodeType )
            {return m_nodeWeightsMap[nodeType]; }
        inline std::vector<Eigen::MatrixXi>& getCertainEdgeWeightsMap( CEdgeTypePtr edgeType )
            {return m_edgeWeightsMap[edgeType]; }
        inline std::vector<std::map<size_t,size_t> >& getGroundTruth(){ return m_groundTruth; }

        inline void addGraph( CGraph &graph ){ m_graphs.push_back( graph ); }

        void addNodeType( CNodeTypePtr nodeType )
        {
            // Add the node type to the vector
            m_nodeTypes.push_back( nodeType );
        }

        void addEdgeType( CEdgeTypePtr edgeType )
        {
            // Add the edge type to the vector of edge types
            m_edgeTypes.push_back( edgeType );
            Eigen::VectorXi defaultTypeOfEdgeFeats( edgeType->getWeights().size() );
            defaultTypeOfEdgeFeats.fill(0);
            m_typesOfEdgeFeatures[ edgeType ] = defaultTypeOfEdgeFeats;
        }

        void addEdgeType( CEdgeTypePtr edgeType, Eigen::VectorXi &typeOfEdgeFeatures )
        {
            // Consistency checks
            assert( edgeType->getNumberOfFeatures() == typeOfEdgeFeatures.rows() );

            ITERATE_SIZE_T(typeOfEdgeFeatures)
                assert( typeOfEdgeFeatures(i) <= 3 );

            // Add the edge type to the vector of edge types
            m_edgeTypes.push_back( edgeType );
            m_typesOfEdgeFeatures[ edgeType ] = typeOfEdgeFeatures;
        }

        void addEdgeType( CEdgeTypePtr edgeType, std::vector<int> typeOfEdgeFeatures  )
        {
            // Add the edge type to the vector of edge types
            m_edgeTypes.push_back( edgeType );
            m_typesOfEdgeFeatures[ edgeType ] = vectorToIntEigenVector( typeOfEdgeFeatures );
        }

        void addClassesRelevance( CNodeTypePtr nodeType, Eigen::VectorXd &classesRelevance )
        {
            m_classesRelevance[nodeType] = classesRelevance;
        }

        /** Returns the execution time in seconds. */
        double getExecutionTime(){ return m_executionTime*pow(10,-9); }

        /** Returns the time spent evaluating the current weights. */
        double getEvaluatingTime(){ return m_evaluatingTime*pow(10,-9); }


        int train( const bool debug = false);

        void updatePseudolikelihood( CGraph &graph,
                                     std::map<size_t,size_t> &groundTruth,
                                     double &fx,
                                     const double *x,
                                     double *g );

        void updateScoreMatching( CGraph &graph,
                                  std::map<size_t,size_t> &groundTruth,
                                  double &fx,
                                  const double *x,
                                  double *g );

        void updatePicewise( CGraph &graph,
                             std::map<size_t,size_t> &groundTruth,
                             double &fx,
                             const double *x,
                             double *g );

        void updateInference( CGraph &graph,
                              std::map<size_t,size_t> &groundTruth,
                              double &fx,
                              const double *x,
                              double *g );

        void updateDecoding( CGraph &graph,
                        std::map<size_t,size_t> &groundTruth,
                        double &fx,
                        const double *x,
                        double *g );




    };
}

#endif
