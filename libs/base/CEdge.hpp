
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

        CNodePtr m_n1;      //!< First node in the edge. It will be always the one with the lower ID.
        CNodePtr m_n2;      //!< Second edge's node.
        Eigen::VectorXd	m_features;     //!< Vector of extracted features for this edge.
        Eigen::MatrixXd	m_potentials;   //!< Computed potentials. Initially empty.
        CEdgeTypePtr       m_type;  //!< Pointer to the edge type.
        size_t          m_id;       //!< ID of the edge. It is unique and automatically assigned.

        size_t setID() { static size_t ID = 0; return ID++; }

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
               Eigen::VectorXd &features ) : m_type(type)
        {            
            m_id = setID();

            // Check that features are a column vector, and transpose it  otherwise
            ( features.cols() > 1 ) ?
                        m_features = features
                      : m_features = features.transpose();

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
                if ( n1->getType()->getID() > n2->getType()->getID() )
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

        /** Additional constructor.
         */
        CEdge( CNodePtr &n1,
               CNodePtr    &n2,
               CEdgeTypePtr   type,
               Eigen::VectorXd &features,
               Eigen::MatrixXd &potentials) : m_potentials(potentials), m_type(type)
        {

            m_id = setID();

            // Check that features are a column vector, and transpose it  otherwise
            ( features.cols() > 1 ) ?
                        m_features = features
                      : m_features = features.transpose();


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
                if ( n1->getType()->getID() > n2->getType()->getID() )
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


        /**	Function for retrieving the ID of the edge.
          * \return The ID of the edge.
          */
        inline size_t getID(){ return m_id; }


        inline CEdgeTypePtr getType() const { return m_type; }


        inline void getNodes ( CNodePtr &n1, CNodePtr &n2 ) const { n1 = m_n1; n2 = m_n2; }

        inline size_t getNodePosition ( size_t ID )
        {
            if ( m_n1->getID() == ID )
                return 0;
            else
                return 1;
        }

        inline size_t getFirstNodeID () const { return m_n1->getID(); }

        inline size_t getSecondNodeID () const { return m_n2->getID(); }

        void getNodesID( size_t &ID1, size_t &ID2 ){ ID1= m_n1->getID(); ID2 = m_n2->getID(); }


        inline Eigen::VectorXd getFeatures() const { return m_features; }

        inline Eigen::VectorXd& getFeatures(){ return m_features; }

        inline void setFeatures( const Eigen::VectorXd &features)
        {
            // Check that features are a column vector, and transpose it  otherwise
            ( features.cols() > 1 ) ?
                        m_features = features
                      : m_features = features.transpose();

        }


        inline Eigen::MatrixXd getPotentials() const { return m_potentials; }

        inline void setPotentials ( Eigen::MatrixXd &potentials ) { m_potentials = potentials; }

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

        friend std::ostream& operator<<(std::ostream& output, const CEdge& e)
        {
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
            ar & m_n1;
            ar & m_n2;
            ar << m_features;
            ar << m_potentials;
            ar & m_type;
            ar & m_id;
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_n1;
            ar & m_n2;
            ar >> m_features;
            ar >> m_potentials;
            ar & m_type;
            ar & m_id;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CEdge, 1)


#endif
