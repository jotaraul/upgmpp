
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
#include "base_utils.hpp"

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
        /**
          * This class defines the structure and common functions for a graph
          * representing an Undirected Probabilistic Graphical Model.
          **/
    private:

        std::multimap<size_t,CEdgePtr>   m_edges_f; //!< Vectof of fast access to all the edges where a node appears.
        std::vector<CEdgePtr>            m_edges;   //!< Vector of graph edges.
        std::vector<CNodePtr>            m_nodes;   //!< Vector of graph nodes.
        size_t                           m_id;      //!< Graph ID.

        /** Private function for obtaning the ID of a new graph.
         */
        size_t setID() const { static size_t ID = 0; return ID++; }

    public:

        /** Default constructor */
        CGraph() { m_id = setID(); }

        /** Function for setting the ID of the graph.
         * \param id: New graph ID.
         */
        inline void setID( size_t id ) { m_id = id; }

        /** Function for getting the graph ID.
         * \return A copy of the graph ID.
         */
        inline size_t getID(){ return m_id; }

        /** Function for getting the graph ID.
         * \return A copy of the graph ID.
         */
        inline size_t getID() const { return m_id; }

        /** Function for adding a node to the graph.
         * \param node: Smart pointer to the node to add.
         */
        inline void     addNode( CNodePtr node ) { m_nodes.push_back(node); }        

        /** Get the graph node laying in a certain position of the vector of nodes.
         * \param index: Node position in that vector.
         * \return A copy of the smart pointer laying in that position.
         */
        inline CNodePtr getNode( size_t index ) { return m_nodes[index]; }

        /** This method allows the user to get a copy of the graph's nodes.
         *  \param v_nodes: Vector of nodes to be filled.
         */
        inline void     getNodes( std::vector<CNodePtr> &v_nodes ) const { v_nodes = m_nodes; }

        /** Get the number of neighbors that a node has.
         * \param nodeID: ID of the node.
         * \return The number of node neighbors.
         */
        inline size_t   getNumberOfNodeNeighbors( size_t nodeID ){ return m_edges_f.count( nodeID ); }

        /** Get a reference to the vector of nodes. The user has to take care
         * using this function, since he/her unintentionally could change it.
         * \return A reference to the vector of nodes.
         */
        inline std::vector<CNodePtr>& getNodes(  ) { return m_nodes; }

        /** Get a reference to the map of nodes and their edges. The user has to take care
         * using this function, since he/her unintentionally could change it.
         * \return A reference to the map of nodes and their edges.
         */
        inline std::multimap<size_t,CEdgePtr>& getEdgesF(){ return m_edges_f; }

        /** Get a copy of the node with a given ID.
         * \param ID: node ID.
         * \return Copy of the node.
         */
        CNodePtr getNodeWithID( size_t ID )
        {
            std::vector<CNodePtr>::iterator it;

            for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
                if ( it->get()->getID() == ID )
                    return *it;

            throw std::logic_error( "Unknown node!" );
        }

        /** Get the vector of edges.
         * \param v_edges: vector to copy the edge smart pointers.
         */
        inline void getEdges( std::vector<CEdgePtr> &v_edges ) const { v_edges = m_edges; }

        /** Get the vector of edges.
         * \return v_edges: a copy of the vector of edge smart pointers.
         */
        inline std::vector<CEdgePtr> getEdges() { return m_edges; }

        /** Get a certain edge given its position in the edges vector.
         * \param index: Position in the edges vector.
         * \return Smart pointer to the edge.
         */
        inline CEdgePtr getEdge( size_t index ){ return m_edges[index]; }

        /** Get the index of an edge in the edges vector given its ID.
         * \param edgeID: ID of the edge.
         * \return Position of that object in the edges vector.
         */
        size_t getEdgeIndex ( size_t edgeID )
        {
            for ( size_t i = 0; i < m_edges.size(); i++ )
                if ( edgeID == m_edges[i]->getID() )
                    return i;

            return -1;
        }

        /** Add an edge to the graph.
         * \param edge: Smart pointer to the edge.
         */
        void addEdge( CEdgePtr edge )
        {

            CNodePtr n1, n2;
            edge->getNodes(n1,n2);
            size_t n1_id = n1->getID();
            size_t n2_id = n2->getID();

            m_edges.push_back(edge);

            m_edges_f.insert( std::pair<size_t, CEdgePtr> (n1_id,edge) );
            m_edges_f.insert( std::pair<size_t, CEdgePtr> (n2_id,edge) );
        }

        /** Delete a node from the graph. It could also produce the deletion of
         * its associated edges.
         * \param ID: ID of the node to delete from the graph.
         */
        void deleteNode( size_t ID )
        {
            // Remove from the vector of nodes

            std::vector<CNodePtr>::iterator it;

            for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
            {
                CNodePtr nodePtr = *it;

                if ( nodePtr->getID() == ID )
                    break;
            }

            if ( it != m_nodes.end() )
                m_nodes.erase( it );

            // Remove edges than contain that node from m_edges and m_edges_f

            std::vector<CEdgePtr>::iterator it2;

            for ( it2 = m_edges.end(); it2 != m_edges.begin(); it2-- )
            {
                CEdgePtr edgePtr = *it2;
                size_t ID1, ID2;

                edgePtr->getNodesID( ID1, ID2 );

                if ( ( ID1 == ID ) || ( ID2 == ID )  )
                    deleteEdge( edgePtr->getID() );
            }

        }

        /** Delete and edge from the graph.
         * \param ID: ID of the edge.
         */
        void deleteEdge( size_t ID )
        {
            CEdgePtr edgePtr;

            std::vector<CEdgePtr>::iterator it;
            for ( it = m_edges.begin(); it != m_edges.end(); it++ )
            {
                if ( (*it)->getID() == ID )
                {
                    edgePtr = *it;
                    break;
                }
            }

            if ( it != m_edges.end() )
            {
                // Delete the edge from the m_edge_f multimap

                size_t ID1, ID2;
                edgePtr->getNodesID(ID1,ID2);

                std::multimap<size_t,CEdgePtr>::iterator it_n1= m_edges_f.find( ID1 );

                for ( ; it_n1 != m_edges_f.end(); it_n1++ )
                {
                    if ( (it_n1->second)->getID() == ID )
                    {
                        m_edges_f.erase( it_n1 );
                        break;
                    }
                }

                std::multimap<size_t,CEdgePtr>::iterator it_n2= m_edges_f.find( ID2 );

                for ( ; it_n2 != m_edges_f.end(); it_n2++ )
                {
                    if ( (it_n2->second)->getID() == ID )
                    {
                        m_edges_f.erase( it_n2 );
                        break;
                    }
                }


                // Delete the edge from the edges vector
                m_edges.erase( it );
            }
        }

        /** Method for dumping a graph to a stream.
         * \param output: output stream.
         * \param g: graph to dump.
         * \return A reference to the output stream.
         */
        friend std::ostream& operator<<(std::ostream& output, const CGraph& g)
        {
            output << "GRAPH ID: " << g.getID() << std::endl;

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

        /** Function for computing the potentials of the nodes and the edges in
         * the graph.
         */
        void computePotentials()
        {
            // Method steps:
            //  1. Compute node potentials
            //  2. Compute edge potentials

            //
            //  1. Node potentials
            //

            std::vector<CNodePtr>::iterator it;

            //cout << "NODE POTENTIALS" << endl;

            for ( it = m_nodes.begin(); it != m_nodes.end(); it++ )
            {
                CNodePtr nodePtr = *it;

                if ( !nodePtr->finalPotentials() )
                {
                    // Get the node type
                    //size_t type = nodePtr->getType()->getID();

                    // Compute the node potentials according to the node type and its
                    // extracted features

                    Eigen::VectorXd potentials = nodePtr->getType()->computePotentials( nodePtr->getFeatures() );

                    // Apply the node class multipliers
                    potentials = potentials.cwiseProduct( nodePtr->getClassMultipliers() );

                    /*Eigen::VectorXd fixed = nodePtr->getFixed();

                    potentials = potentials.cwiseProduct( fixed );*/

                    nodePtr->setPotentials( potentials );
                }

            }

            //
            //  2. Edge potentials
            //

            std::vector<CEdgePtr>::iterator it2;

            //cout << "EDGE POTENTIALS" << endl;

            for ( it2 = m_edges.begin(); it2 != m_edges.end(); it2++ )
            {
                CEdgePtr edgePtr = *it2;

                Eigen::MatrixXd potentials
                        = edgePtr->getType()->computePotentials( edgePtr->getFeatures() );

                edgePtr->setPotentials ( potentials );
            }

        }

        /** This method permits the user to get the unnormalized log likelihood
         * of the graph given a certain assignation to all its nodes.
         * \param classes: Classes assignation to all the nodes.
         * \return Unnormalized log likelihood.
         */
        double getUnnormalizedLogLikelihood( std::map<size_t,size_t> &classes )
        {
            double unlikelihood = 1;

            //size_t N_nodes = m_nodes.size();
            size_t N_edges = m_edges.size();

            std::map<size_t,size_t>::iterator it;

            for ( it = classes.begin(); it != classes.end(); it++ )
            {
                CNodePtr node = getNodeWithID( it->first );
                unlikelihood *= node->getPotentials()(classes[node->getID()]);
            }

            for ( size_t index = 0; index < N_edges; index++ )
            {
                CEdgePtr edge = m_edges[index];
                CNodePtr n1, n2;
                edge->getNodes(n1,n2);
                size_t ID1 = n1->getID();
                size_t ID2 = n2->getID();

                if ( ID1 > ID2 )
                    unlikelihood *= edge->getPotentials()(classes[ID2],classes[ID1]);
                else
                    unlikelihood *= edge->getPotentials()(classes[ID1],classes[ID2]);

            }

            unlikelihood = std::log( unlikelihood );

            return unlikelihood;
        }

        //
        //  Serialization stuff
        //
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            // note, version is always the latest when saving
            ar & m_id;
            ar & m_nodes;
            ar & m_edges;

            size_t N_fields = m_edges_f.size();

            ar & N_fields;

            std::multimap<size_t,CEdgePtr>::const_iterator it;
            for ( it = m_edges_f.begin(); it != m_edges_f.end(); it++ )
            {
                ar & it->first;
                ar & it->second;
            }
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_id;
            ar & m_nodes;
            ar & m_edges;

            size_t N_fields;

            ar & N_fields;

            for ( size_t i = 0; i < N_fields; i++ )
            {
                size_t ID;
                CEdgePtr edgePtr;

                ar & ID;
                ar & edgePtr;

                m_edges_f.insert( std::pair<size_t, CEdgePtr> ( ID , edgePtr) );
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CGraph, 1)


#endif

