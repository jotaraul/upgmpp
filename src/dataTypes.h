
#ifndef GCRF_TYPES
#define GCRF_TYPES

#include <string>
#include <vector>
#include <map>
#include <list>
#include <Eigen/Dense>

#include <iostream>

namespace PGMplusplus
{
	class CNode
	{
	private:
		size_t 		m_id;		// Id of the node. Must be unique.
		size_t		m_type;		// Type of the node.
		std::string	m_label;	// Label of the node. If not needed, just for human-readalbe purpouses.
        Eigen::VectorXd	m_features;	// Features of that node.
        Eigen::VectorXd	m_potentials; 	// Potentials computed for this node.

	public:
		/*
		 *	Defaul constructor
		 */
        CNode( )
        {
            static size_t ID = 0;
            m_id = ID;
            ID++;
        }

        CNode( size_t type,
            Eigen::VectorXd &features,
            std::string label="" ) : m_type(type), m_label(label), m_features(features)
        {
            static size_t ID = 0;
            m_id = ID;
            ID++;
        }


		/*
		 *	Defaul destructor
		 */
		~CNode(){}
 
		/*
		 *	Function for accessing/modifying member data
		 */
        inline size_t getId() const { return m_id; }
        inline size_t getType()	const { return m_type; }
        inline std::string getLabel() const { return m_label; }
        inline Eigen::VectorXd getFeatures() const { return m_features;	}
        inline Eigen::VectorXd getPotentials() const { return m_potentials; }
        inline void setPotentials( const Eigen::VectorXd &pot ){ m_potentials = pot; }

        size_t getMostProbableClass();

        friend std::ostream& operator<<(std::ostream& output, const CNode& n)
        {
            output << "ID: " << n.getId() << std::endl;
            output << "Label: " << n.getLabel() << std::endl;
            output << "Features: " << n.getFeatures() << std::endl;
            output << "Potentials: " << n.getPotentials() << std::endl;

            return output;
        }

	};

	class CEdge
	{
	private:

		CNode m_n1;
        CNode m_n2; // Change for ids
        Eigen::VectorXd	m_features;
        Eigen::MatrixXd	m_potentials;
        size_t          m_type;

	public:	

        CEdge() {}

        CEdge( CNode &n1,
               CNode &n2,
               size_t type) : m_type(type)
        {
            if ( n1.getType() > n2.getType() )
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
            size_t   type,
            Eigen::VectorXd &features,
            Eigen::MatrixXd &potentials) : m_features(features), m_potentials(potentials), m_type(type)
		{
			if ( n1.getType() > n2.getType() )
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

        inline size_t getType() const { return m_type; }
        inline void getNodes ( CNode &n1, CNode &n2 ) const { n1 = m_n1; n2 = m_n2; }
        inline size_t getSecondNodeID () const { return m_n2.getId(); }
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

    class CGraph
    {
    private:

        std::vector<std::vector<size_t> >   m_edges_f;
        std::vector<CEdge>                  m_edges;
        std::vector<CNode>                  m_nodes;
        std::vector<Eigen::MatrixXd>        m_nodeWeights;
        std::vector<std::vector<Eigen::MatrixXd> > m_edgeWeights;

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

        void addNodeWeights( Eigen::MatrixXd &weights )
        {
            size_t N_elements = m_nodeWeights.size();
            m_nodeWeights.resize( N_elements + 1 );

            m_nodeWeights[ N_elements ] = weights;
        }

        void addEdgeWeights( std::vector<Eigen::MatrixXd> &weights )
        {
            size_t N_elements = m_edgeWeights.size();
            m_edgeWeights.resize( N_elements + 1 );

            m_edgeWeights[ N_elements ] = weights;
        }

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
}  

#endif
