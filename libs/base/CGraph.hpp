
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

#ifndef _UPGMpp_GRAPH_
#define _UPGMpp_GRAPH_


#include "CNodeType.hpp"
#include "CEdgeType.hpp"
#include "CNode.hpp"
#include "CEdge.hpp"

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

#include <iostream>
#include <boost/shared_ptr.hpp>

namespace UPGMpp
{

    class CGraph
    {
    private:

        std::multimap<size_t,CEdgePtr>   m_edges_f;
        std::vector<CEdgePtr>            m_edges;
        std::vector<CNodePtr>            m_nodes;

    public:

        CGraph() {}

        inline void     addNode( CNodePtr node ) { m_nodes.push_back(node); }
        inline CNodePtr getNode( size_t index ) { return m_nodes[index]; }
        inline void     getNodes( std::vector<CNodePtr> &v_nodes ) const { v_nodes = m_nodes; }
        inline std::vector<CNodePtr>& getNodes(  ) { return m_nodes; }
        inline std::multimap<size_t,CEdgePtr>& getEdgesF(){ return m_edges_f; }

        CNodePtr getNodeWithID( size_t id )
        {
            std::vector<CNodePtr>::iterator it;

            for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
                if ( it->get()->getId() == id )
                    return *it;
        }

        inline void getEdges( std::vector<CEdgePtr> &v_edges ) const { v_edges = m_edges; }
        inline CEdgePtr getEdge( size_t index ){ return m_edges[index]; }

        void addEdge( CEdgePtr edge )
        {

            CNodePtr n1, n2;
            edge->getNodes(n1,n2);
            size_t n1_id = n1->getId();
            size_t n2_id = n2->getId();

            m_edges.push_back(edge);
            size_t size = m_edges.size();
            size_t f_size = m_edges_f.size();

            m_edges_f.insert( std::pair<size_t, CEdgePtr> (n1_id,edge) );
            m_edges_f.insert( std::pair<size_t, CEdgePtr> (n2_id,edge) );
        }

        void deleteNode( size_t id )
        {
            // Remove from the vector of nodes

            std::vector<CNodePtr>::iterator it;

            for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
            {
                CNodePtr nodePtr = *it;

                if ( nodePtr->getId() == id )
                    break;
            }

            if ( it != m_nodes.end() )
                m_nodes.erase( it );

            // Remove edges than contain that node

            std::vector<CEdgePtr>::iterator it2;

            for ( it2 = m_edges.end(); it2 != m_edges.begin(); it2-- )
            {
                CEdgePtr edgePtr = *it2;
                size_t ID1, ID2;

                edgePtr->getNodesID( ID1, ID2 );

                if ( ( ID1 == id ) || ( ID2 == id )  )
                    m_edges.erase( it2 );
            }

            // TODO: Remove edges from fast access structure

        }

        friend std::ostream& operator<<(std::ostream& output, const CGraph& g)
        {
            std::vector<CNodePtr> v_nodes;
            g.getNodes(v_nodes);

            output << "NODES" << std::endl;

            for ( size_t i = 0; i < v_nodes.size(); i++ )
                output << *v_nodes[i] << std::endl ;

            std::vector<CEdgePtr> v_edges;
            g.getEdges(v_edges);

            output << "EDGES" << std::endl;

            for ( size_t i = 0; i < v_edges.size(); i++ )
                output << *v_edges[i] << std::endl ;

            return output;
        }

        void computePotentials();

        double getUnnormalizedLogLikelihood( std::map<size_t,size_t> &classes );
    };

}

#endif

