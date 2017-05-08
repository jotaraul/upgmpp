
/*---------------------------------------------------------------------------*
 |                               UPGM++                                      |
 |                   Undirected Graphical Models in C++                      |
 |                                                                           |
 |          Copyright (C) 2014-2017 Jose Raul Ruiz Sarmiento                 |
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

#ifndef _UPGMpp_EDGE_
#define _UPGMpp_EDGE_

#include "base_utils.hpp"

#include <string>
#include <Eigen/Dense>

#include <iostream>
#include <boost/shared_ptr.hpp>


namespace UPGMpp
{

    /** Anticipated class declararions */
    class CEdge;

    /** Useful type definitions */
    typedef boost::shared_ptr<CEdge> CEdgePtr;

    class CEdge
    {
    private:

        CNodePtr        m_n1;      //!< First node in the edge. It will be always the one with the lower ID.
        CNodePtr        m_n2;      //!< Second edge's node.
        Eigen::VectorXd	m_features;     //!< Vector of extracted features for this edge.
        Eigen::MatrixXd	m_potentials;   //!< Computed potentials. Initially empty.
        CEdgeTypePtr    m_type;  //!< Pointer to the edge type.
        size_t          m_id;       //!< ID of the edge. It is unique and automatically assigned.
        bool            m_handCodedPotentials;

        /** Private function for obtaning the ID of a new edge.
         */
        size_t setID() { static size_t ID = 0; return ID++; }

        /** Private funtion for properly setting the first and second edge nodes.
         */
        void setNodeTypes( CNodePtr n1, CNodePtr n2, CEdgeTypePtr type )
        {
            // The first node is always the one with the lower type of ID node.
            // If they share the same type of node, then the first node is
            // always the one with the lower node ID.
            if ( n1->getType()->getID() != n2->getType()->getID() )
            {
                if ( n1->getType()->getID() == type->getN1Type()->getID() )
                {
                    m_n1 = n1;
                    m_n2 = n2;
                }
                else
                {
                    m_n1 = n2;
                    m_n2 = n1;
                }
            }
            else
            {
                if ( n1->getID() > n2->getID() )
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
        }

    public:

        /** Default constructor.
         */
        CEdge()
        {
            m_id = setID();
        }

        /** Additional constructor.
         */
        CEdge( CNodePtr n1,
               CNodePtr n2,
               CEdgeTypePtr type,
               Eigen::VectorXd &features ) : m_features( features ),
                                             m_type(type)

        {            
            m_id = setID();

            setNodeTypes( n1, n2, type );
        }

        /** Additional constructor.
         */
        template <typename T> CEdge( CNodePtr n1,
               CNodePtr n2,
               CEdgeTypePtr type,
               T &features ) : m_type(type)
        {

            m_features =  vectorToEigenVector( features );

            m_id = setID();

            setNodeTypes( n1, n2, type );

        }

        /** Additional constructor.
         */
        CEdge( CNodePtr &n1,
               CNodePtr    &n2,
               CEdgeTypePtr   type,
               Eigen::VectorXd &features,
               Eigen::MatrixXd &potentials) : m_features( features ),
                                              m_potentials(potentials),
                                              m_type(type)

        {
            m_id = setID();

            setNodeTypes( n1, n2, type );
        }


        /**	Function for retrieving the ID of the edge.
          * \return The ID of the edge.
          */
        inline size_t getID(){ return m_id; }

        /**	Function for retrieving the ID of the edge.
          * \return The ID of the edge.
          */
        inline size_t getID() const { return m_id; }

        /** Function for setting the edge ID.
         * \param ID: new edge ID.
         */
        inline void setID( size_t ID ){ m_id = ID; }

        /** Function for getting the edge type.
         * \return A copy of the edge
         */
        inline CEdgeTypePtr getType() const { return m_type; }

        /** Function for setting the edge type.
          * \param edgeTypePtr: new edge type.
          */
        inline void setType( CEdgeTypePtr edgeTypePtr){ m_type = edgeTypePtr; }

        /** Function for retrieving the two nodes linked by the edge.
         * \param n1: First node of the edge.
         * \param n2: Second node of the edge.
         */
        inline void getNodes ( CNodePtr &n1, CNodePtr &n2 ) const { n1 = m_n1; n2 = m_n2; }

        /** This method allows the user to ask about the position of a node in
         * the edge. Is this the first or the second node?
         * \param ID: ID of the node.
         * \return 0 if it is the first node, 1 otherwise.
         */
        inline size_t getNodePosition ( size_t ID )
        {
            if ( m_n1->getID() == ID )
                return 0;
            else
                return 1;
        }

        /** Get the ID of the first node.
         * \return ID of the fist node.
         */
        inline size_t getFirstNodeID () const { return m_n1->getID(); }

        /** Get the ID of the second node.
         * \return ID of the second node.
         */
        inline size_t getSecondNodeID () const { return m_n2->getID(); }

        /** Get the ID of both nodes.
         * \param ID1: ID of the first node.
         * \param ID2: ID of the second node.
         */
        void getNodesID( size_t &ID1, size_t &ID2 ){ ID1= m_n1->getID(); ID2 = m_n2->getID(); }

        /** This method permits the users to get the edge features.
         * \return A copy of the vector of edge features.
         */
        inline Eigen::VectorXd getFeatures() const { return m_features; }

        /** This method permits the users to get a reference to the edge features.
         * Take care with the reference! You could unintentionally change the
         * vector values from your code.
         * \return A reference to the vector of edge features.
         */
        inline Eigen::VectorXd& getFeatures(){ return m_features; }

        /** Set the edge features.
         * \param features: Vector of features.
         */
        inline void setFeatures( const Eigen::VectorXd &features)
        {
            m_features = features;
        }

        /** Set the edge features.
         * \param features: Vector of features.
         */
        template<typename T> inline void setFeatures( const T &features)
        {
            m_features = vectorToEigenVector( features );
        }

        /** Get a copy of the edge potentials.
         * \return A copy of the edge potentials.
         */
        inline Eigen::MatrixXd getPotentials() const { return m_potentials; }

        /** Get a reference to the edge potentials.
         * \return A reference to the edge potentials.
         */
        inline Eigen::MatrixXd & getPotentials() { return m_potentials; }

        /** Set the edge potentials.
         * \param potentials: Matrix of new edge potentials.
         */
        inline void setPotentials ( Eigen::MatrixXd &potentials ) { m_potentials = potentials; }

        /** Set the edege potentials and make them final, so they will be not
         * computed again depending on the features. This permits the user to
         * work with Markov Random Fields in addition to Conditional Random
         * Fields.
         * \param potentials: Matrix of final edge potentials.
         */
        void setFinalPotentials( Eigen::MatrixXd &potentials )
        {
            m_potentials = potentials;
            m_handCodedPotentials = true;
        }

        /** Returns if the potentials of the edge were set manually and then are final.
         *  \return True if the edge potentials were set manually, false
         *           otherwise.
         */
        inline bool finalPotentials(){ return m_handCodedPotentials; }

        /** If the value of a node is fixed, get the edge potentials for the
         * neighbor node considering that fixed value.
         * \param nodeID: ID of the node.
         * \param neighborClass: fixed value.
         */
        Eigen::VectorXd getNeighborPotentialsForNodeFixedValue( size_t nodeID, size_t neighborClass )
        {
            if ( nodeID == m_n1->getID() )
            {
                return m_potentials.col( neighborClass );
            }
            else
            {
                return m_potentials.row( neighborClass ).transpose();
            }
        }

        /**	Function for prompting the content of an edge.
          * \return An stream with the edge information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CEdge& e)
        {
            output << "Edge ID:" << e.getID() << std::endl;
            output << "Type: " << e.getType()->getLabel() << std::endl;
            CNodePtr n1, n2;
            e.getNodes( n1, n2);

            output << "n1 ID:" << n1->getID() << std::endl ;
            output << "n2 ID:" << n2->getID() << std::endl ;

            output << "features:" << e.getFeatures() << std::endl ;
            output << "potentials:" << e.getPotentials() << std::endl ;


            return output;
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
            ar & m_type;
            ar & m_n1;
            ar & m_n2;
            ar << m_features;
            ar << m_potentials;
            ar << m_handCodedPotentials;

        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_id;
            ar & m_type;
            ar & m_n1;
            ar & m_n2;
            ar >> m_features;
            ar >> m_potentials;
            ar >> m_handCodedPotentials;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CEdge, 1)


#endif
