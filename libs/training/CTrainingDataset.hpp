
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


    class CTrainingDataSet
    {
    private:
        std::vector<CGraph> m_graphs;
        std::vector<std::map<size_t,size_t> >      m_groundTruth;
        std::vector<CNodeTypePtr>    m_nodeTypes;
        std::vector<CEdgeTypePtr>    m_edgeTypes;
        size_t                       N_weights;
        std::map<CNodeTypePtr, Eigen::MatrixXi>             m_nodeWeightsMap;
        std::map<CEdgeTypePtr,std::vector<Eigen::MatrixXi> > m_edgeWeightsMap;

    public:

        CTrainingDataSet(){}

        inline void setGroundTruth( std::vector<std::map<size_t,size_t> > &gt ) { m_groundTruth = gt; }
        inline void addGraphGroundTruth( std::map<size_t,size_t> &graph_gt ){ m_groundTruth.push_back( graph_gt ); }

        inline std::vector<CGraph>& getGraphs(){ return m_graphs; }
        inline std::vector<CNodeTypePtr>& getNodeTypes(){ return m_nodeTypes; }
        inline std::vector<CEdgeTypePtr>& getEdgeTypes(){ return m_edgeTypes; }
        inline Eigen::MatrixXi& getCertainNodeWeightsMap( CNodeTypePtr nodeType )
            {return m_nodeWeightsMap[nodeType]; }
        inline std::vector<Eigen::MatrixXi>& getCertainEdgeWeightsMap( CEdgeTypePtr edgeType )
            {return m_edgeWeightsMap[edgeType]; }
        inline std::vector<std::map<size_t,size_t> >& getGroundTruth(){ return m_groundTruth; }

        inline void addGraph( CGraph graph ){ m_graphs.push_back( graph ); }

        void addNodeType( CNodeTypePtr nodeType )
        {
            // Add the node type to the vector
            m_nodeTypes.push_back( nodeType );
        }

        void addEdgeType( CEdgeTypePtr edgeType )
        {
            // Add the edge type to the vector of edge types
            m_edgeTypes.push_back( edgeType );
        }

        void train();

        void updateFunctionValueAndGradients( CGraph &graph,
                                            std::map<size_t,size_t> &groundTruth,
                                            double &fx,
                                            const double *x,
                                            double *g );



    };
}

#endif
