
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


#ifndef GCRF_TYPES
#define GCRF_TYPES

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <iostream>

namespace UPGMplusplus
{

    class CNodeType
    {
    private:
        Eigen::MatrixXi m_weights_map;
        Eigen::MatrixXd m_weights;
        size_t          m_ID;
        std::string     m_label;

    public:

        CNodeType()
        {
            static size_t ID = 0;
            m_ID = ID;
            ID++;
        }

        inline size_t& getID() { return m_ID; }
        inline void setLabel( std::string &label ) { m_label = label; }
        inline void setWeightsMap( Eigen::MatrixXi &w_m ){ m_weights_map = w_m; }
        inline void setWeights( Eigen::MatrixXd &weights ) { m_weights = weights; }
        inline Eigen::MatrixXd& getWeights(){ return m_weights; }

    };

    class CEdgeType
    {
    private:
        std::vector<Eigen::MatrixXi> m_weights_map;
        std::vector<Eigen::MatrixXd> m_weights;
        size_t      m_ID;
        std::string m_label;
    public:
        CEdgeType()
        {
            static size_t ID = 0;
            m_ID = ID;
            ID++;
        }

        inline size_t& getID() { return m_ID; }
        inline void setLabel( std::string &label ) { m_label = label; }
        inline void setWeightsMap( std::vector<Eigen::MatrixXi> &w_m ){ m_weights_map = w_m; }
        inline void setWeights( std::vector<Eigen::MatrixXd> &weights ) { m_weights = weights; }
        inline std::vector<Eigen::MatrixXd>& getWeights() { return m_weights; }
    };


/*----------------------------------------------------------------------------
 *
 *                                 CNode
 *
 *---------------------------------------------------------------------------*/
	class CNode
	{
        /**
          * This class defines the structure and common functions for the nodes
          * of an Undirected Probabilistic Model (UPGM).
          **/
	private:
        size_t 		m_id;		//!< Id of the node. Must be unique. This is ensured in the constructor and is transparent for the user.
        CNodeType   m_type;		//!< Type of the node.
        std::string	m_label;	//!< Label of the node. If not needed, just for human-readalbe purpouses.
        Eigen::VectorXd	m_features;	//!< Features extracted for this node.
        Eigen::VectorXd	m_potentials; 	//!< Potentials computed for this node. They can come from the user, or for the computePotentials function in the Graph class.

	public:

        /** Default constructor
          */
        CNode( )
        {
            static size_t ID = 0;
            m_id = ID;
            ID++;
        }

        /** Addtional constructor
          */
        CNode( CNodeType type,
            Eigen::VectorXd &features,
            std::string label="" ) : m_type(type),
                                    m_label(label),
                                    m_features(features)
        {
            static size_t ID = 0;
            m_id = ID;
            ID++;
        }


        /**	 Defaul destructor
          */
		~CNode(){}

        /**	Function for retrieving the ID of the node.
          * \return The ID of the node.
          */
        inline size_t getId() const { return m_id; }

        /**	Function for retrieving the node's type.
          * \return The node's type.
          */
        inline CNodeType getType()	const { return m_type; }

        /**	Function for retrieving the Label of the node.
          * \return The Label of the node.
          */
        inline std::string getLabel() const { return m_label; }

        /**	Function for retrieving the features of the node.
          * \return The features of the node.
          */
        inline Eigen::VectorXd getFeatures() const { return m_features;	}

        /**	Function for setting the node's features
          * \param feat: the new features of the node.
          */
        inline void setFeatures( Eigen::VectorXd &feat) { m_features = feat; }

        /**	Function for retrieving the potentials of the node.
          * \return The potentials of the node.
          */
        inline Eigen::VectorXd getPotentials() const { return m_potentials; }

        /**	Function for setting the node's potentials
          * \param pot: new node potentials.
          */
        inline void setPotentials( const Eigen::VectorXd &pot ){ m_potentials = pot; }

        /**	Function for prompting the content of a node
          * \return An stream with the node's information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CNode& n)
        {
            output << "ID: " << n.getId() << std::endl;
            output << "Label: " << n.getLabel() << std::endl;
            output << "Features: " << n.getFeatures() << std::endl;
            output << "Potentials: " << n.getPotentials() << std::endl;

            return output;
        }

	};


/*----------------------------------------------------------------------------
 *
 *                                 CEdge
 *
 *---------------------------------------------------------------------------*/

	class CEdge
	{
	private:

		CNode m_n1;
        CNode m_n2; // Change for ids
        Eigen::VectorXd	m_features;
        Eigen::MatrixXd	m_potentials;
        CEdgeType       m_type;

	public:	

        CEdge() {}

        CEdge( CNode &n1,
               CNode &n2,
               CEdgeType type) : m_type(type)
        {
            if ( n1.getType().getID() > n2.getType().getID() )
            {
                m_n1 = n2;
                m_n2 = n1;
            }
            else
            {
                m_n1 = n1;
                m_n2 = n2;
            }
        }

		CEdge( CNode &n1, 
            CNode    &n2,
            CEdgeType   type,
            Eigen::VectorXd &features,
            Eigen::MatrixXd &potentials) : m_features(features), m_potentials(potentials), m_type(type)
		{
            if ( n1.getType().getID() > n2.getType().getID() )
			{
				m_n1 = n2;
				m_n2 = n1;
			}
			else
			{
				m_n1 = n1;
				m_n2 = n2;
			}
		}

        inline CEdgeType getType() const { return m_type; }
        inline void getNodes ( CNode &n1, CNode &n2 ) const { n1 = m_n1; n2 = m_n2; }
        inline size_t getSecondNodeID () const { return m_n2.getId(); }
        void getNodesID( size_t &ID1, size_t &ID2 ){ ID1= m_n1.getId(); ID2 = m_n2.getId(); }
        inline Eigen::VectorXd getFeatures() const { return m_features; }
        inline void setFeatures( const Eigen::VectorXd &features) { m_features = features; }
        inline Eigen::MatrixXd getPotentials() const { return m_potentials; }
        inline void setPotentials ( Eigen::MatrixXd &potentials ) { m_potentials = potentials; }

        friend std::ostream& operator<<(std::ostream& output, const CEdge& e)
        {
            CNode n1, n2;
            e.getNodes( n1, n2);

            output << "n1:" << n1.getId() << std::endl ;
            output << "n2:" << n2.getId() << std::endl ;

            output << "features:" << e.getFeatures() << std::endl ;
            output << "potentials:" << e.getPotentials() << std::endl ;


            return output;
        }

	};

/*----------------------------------------------------------------------------
 *
 *                                 CGraph
 *
 *---------------------------------------------------------------------------*/

    class CGraph
    {
    private:

        std::vector<std::vector<size_t> >   m_edges_f;
        std::vector<CEdge>                  m_edges;
        std::vector<CNode>                  m_nodes;

        struct TOptions
        {
            size_t ICM_maxIterations;
        } m_TOptions;

    public:

        CGraph() {}

        inline void addNode( CNode &node ) { m_nodes.push_back(node); }
        inline CNode& getNode( size_t index ) { return m_nodes[index]; }
        inline void getNodes( std::vector<CNode> &v_nodes ) const { v_nodes = m_nodes; }
        inline void getEdges( std::vector<CEdge> &v_edges ) const { v_edges = m_edges; }
        inline CEdge& getEdge( size_t index ){ return m_edges[index]; }

        void addEdge( CEdge &edge )
        {

            CNode n1, n2;
            edge.getNodes(n1,n2);
            size_t n1_id = n1.getId();
            size_t n2_id = n2.getId();

            m_edges.push_back(edge);
            size_t size = m_edges.size();
            size_t f_size = m_edges_f.size();

            if ( f_size <= n2_id )
                m_edges_f.resize( n2_id + 1 );

            m_edges_f.at(n1_id).push_back(size-1);
            m_edges_f.at(n2_id).push_back(size-1);
        }

        friend std::ostream& operator<<(std::ostream& output, const CGraph& g)
        {
            std::vector<CNode> v_nodes;
            g.getNodes(v_nodes);

            output << "NODES" << std::endl;

            for ( size_t i = 0; i < v_nodes.size(); i++ )
                output << v_nodes[i] << std::endl ;

            std::vector<CEdge> v_edges;
            g.getEdges(v_edges);

            output << "EDGES" << std::endl;

            for ( size_t i = 0; i < v_edges.size(); i++ )
                output << v_edges[i] << std::endl ;

            return output;
        }

        void computePotentials();

        void decodeICM( std::vector<size_t> &results );
        void decodeGreedy( std::vector<size_t> &results );

    };

/*----------------------------------------------------------------------------
 *
 *                            CTrainingDataSet
 *
 *---------------------------------------------------------------------------*/


    class CTrainingDataSet
    {
    private:
        std::vector<CGraph> m_graphs;
        Eigen::VectorXd     m_nodeWeights;
        Eigen::VectorXd     m_edgeWeights;

    public:

        CTrainingDataSet(){}

        inline void addGraph( CGraph graph ){ m_graphs.push_back( graph ); }

        void setNodeWeights( Eigen::VectorXd &nodeWeights )
        {
            m_nodeWeights = nodeWeights;
        }

        void setEdgeWeights( Eigen::VectorXd &edgeWeights )
        {
            m_edgeWeights = edgeWeights;
        }

        void train();


    };
}  

#endif
