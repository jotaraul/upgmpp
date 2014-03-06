
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

#ifndef _UPGMpp_NODE_
#define _UPGMpp_NODE_

#include "base_utils.hpp"

namespace UPGMpp
{
    class CNode;

    typedef boost::shared_ptr<CNode> CNodePtr;

    class CNode
    {
        /**
          * This class defines the structure and common functions for the nodes
          * of an Undirected Probabilistic Model (UPGM).
          **/
    private:
        size_t          m_id;		//!< Id of the node. Must be unique. This is ensured in the constructor and is transparent for the user.
        CNodeTypePtr    m_type;		//!< Type of the node.
        std::string     m_label;	//!< Label of the node. If not needed, just for human-readalbe purpouses.
        Eigen::VectorXd	m_features;	//!< Features extracted for this node, stored as a column vector
        Eigen::VectorXd	m_potentials; 	//!< Potentials computed for this node. They can come from the user, or for the computePotentials function in the Graph class.        

        size_t setID() { static size_t ID = 0; return ID++; }


    public:

        /** Default constructor
          */
        CNode( )
        {
            m_id = setID();
        }

        /** Additional constructor
          */
        CNode( CNodeTypePtr type,
            Eigen::VectorXd &features,
            std::string label="" ) : m_type ( type ),
                                     m_label ( label )
        {
            m_id = setID();

            // Check that features are a column vector, and transpose it  otherwise
            ( features.cols() > 1 ) ?
                        m_features = features
                      : m_features = features.transpose();
        }


        /**	 Defaul destructor
          */
        ~CNode(){}

        /**	Function for retrieving the ID of the node.
          * \return The ID of the node.
          */
        inline size_t getID() const { return m_id; }

        /**	Function for retrieving the node's type.
          * \return The node's type.
          */
        inline CNodeTypePtr getType()	const { return m_type; }

        /**	Function for seting the node's type.
          * \param n_t: The new node's type.
          */
        inline void setType( CNodeTypePtr n_t ){ m_type = n_t; }

        /**	Function for setting the Label of the node.
          * param The new label of the node.
          */
        inline void setLabel( std::string  &label) { m_label = label; }

        /**	Function for retrieving the Label of the node.
          * \return The Label of the node.
          */
        inline std::string getLabel() const { return m_label; }

        /**	Function for retrieving the features of the node.
          * \return The features of the node.
          */
        inline Eigen::VectorXd getFeatures() const { return m_features;	}

        /**	Function for retrieving the features of the node.
          * \return The features of the node.
          */
        inline Eigen::VectorXd& getFeatures(){ return m_features;	}

        /**	Function for setting the node's features
          * \param feat: the new features of the node.
          */
        void setFeatures( Eigen::VectorXd &features)
        {
            // Check that features are a column vector, and transpose it  otherwise
            ( features.cols() > 1 ) ?
                        m_features = features
                      : m_features = features.transpose();
        }

        /**	Function for retrieving the potentials of the node.
          * \return The potentials of the node.
          */
        inline Eigen::VectorXd getPotentials() const { return m_potentials; }

        /**	Function for retrieving the potentials of the node.
          * \return The potentials of the node.
          */
        //inline Eigen::VectorXd& getPotentials() { return m_potentials; }

        /**	Function for setting the node's potentials
          * \param pot: new node potentials.
          */
        inline void setPotentials( const Eigen::VectorXd &pot ){ m_potentials = pot; }

        /**	Function for prompting the content of a node
          * \return An stream with the node's information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CNode& n)
        {
            output << "ID: " << n.getID() << std::endl;
            output << "Label: " << n.getLabel() << std::endl;
            output << "Features: " << n.getFeatures() << std::endl;
            output << "Potentials: " << n.getPotentials() << std::endl;

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
            ar & m_label;
            ar << m_features;
            ar << m_potentials;
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_id;
            ar & m_type;
            ar & m_label;
            ar >> m_features;
            ar >> m_potentials;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CNode, 1)

#endif
