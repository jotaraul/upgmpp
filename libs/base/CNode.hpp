
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
        Eigen::VectorXd m_classMultipliers; //!< A multiplier applied to each class, initially a vector of ones
        Eigen::VectorXd m_fixed;    //!< This node is fixed to a some value?
        bool            m_handCodedPotentials; //!< Were the potentials of the node manually introduced? Useful to no compute them in a computePotentials request from a graph.

        /** Private function for obtaning the ID of a new node.
         */
        size_t setID() { static size_t ID = 0; return ID++; }


    public:

        /** Default constructor.
          */
        CNode( ) : m_handCodedPotentials(false)
        {
            m_id = setID();
        }

        /** Additional constructor.
          */
        CNode( CNodeTypePtr type,
            Eigen::VectorXd &features,
            std::string label="" ) : m_type ( type ),
                                     m_label ( label ),
                                     m_features( features ),
                                     m_handCodedPotentials( false )
        {
            assert( features.rows() == type->getNumberOfFeatures() );

            m_id = setID();

            size_t N_classes = type->getNumberOfClasses();

            m_fixed.resize( N_classes );
            m_fixed.fill(1);

            m_classMultipliers.resize( N_classes );
            m_classMultipliers.fill(1);

        }

        /** Additional constructor.
          */
        template<typename T> CNode(  CNodeTypePtr type,
                                     T &features,
                                     std::string label="" ): m_type ( type ),
                                                             m_label ( label ),
                                                             m_handCodedPotentials( false )
        {
            assert( features.rows() == type->getNumberOfFeatures() );

            m_id = setID();

            // Convert the vector of features into an eigen vector            

            m_features = vectorToEigenVector( features );

            //
            size_t N_classes = type->getNumberOfClasses();

            m_fixed.resize( N_classes );
            m_fixed.fill(1);

            m_classMultipliers.resize( N_classes );
            m_classMultipliers.fill(1);
        }


        /**	Function for retrieving the ID of the node.
          * \return The ID of the node.
          */
        inline size_t getID() const { return m_id; }

        /** CAUTION: The user must take care with this function. It must be used
         * only if he/her is going to take care / assign the IDs to all the nodes
         * in the graph.
         * \param newID: node ID to be set.
         */
        inline void setID( size_t newID ){ m_id = newID; }

        /**	Function for retrieving the node's type.
          * \return The node's type.
          */
        inline CNodeTypePtr getType()	const { return m_type; }

        /**	Function for seting the node's type.
          * \param n_t: The new node's type.
          */
        void setType( CNodeTypePtr n_t )
        {
            m_type = n_t;

            size_t N_classes = n_t->getNumberOfClasses();

            m_fixed.resize( N_classes );
            m_fixed.fill(1);

            m_classMultipliers.resize( N_classes );
            m_classMultipliers.fill(1);
        }

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
            assert( features.rows() == m_type->getNumberOfFeatures() );

            m_features = features;
        }

        /**	Function for setting the node's features
          * \param feat: the new features of the node.
          */
        template<typename T> void setFeatures( T &features)
        {
            assert( features.size() == m_type->getNumberOfFeatures() );

            m_features = vectorToEigenVector( features );
        }

        /** This method fix the node to a certain class.
         * \param toClass: class to employ to fix the node.
         */
        void fix( size_t toClass )
        {
            assert( toClass < m_type->getNumberOfClasses() );

            m_fixed.fill(0);
            m_fixed( toClass ) = 1;
        }

        /**	Function for retrieving the potentials of the node.
          * \return The potentials of the node.
          */
        inline Eigen::VectorXd getPotentials( bool considerFixed = false ) const
        {
            if ( considerFixed )
                return m_potentials.cwiseProduct( m_fixed );
            else
                return m_potentials;
        }

        /**	Function for retrieving the potentials of the node.
          * \return The potentials of the node.
          */
        inline const Eigen::VectorXd getPotentials( bool considerFixed = false )
        {
            if ( considerFixed )
            {
                VectorXd potentials = m_potentials.cwiseProduct( m_fixed );
                return potentials;
            }
            else
                return m_potentials;
        }

        /**	Function for setting the node's potentials
          * \param pot: new node potentials.
          */
        inline void setPotentials( const Eigen::VectorXd &pot )
        {
            assert( pot.rows() == m_type->getNumberOfClasses() );
            m_potentials = pot;
        }

        /** This method allows the user to set the node's potentials to a final
         * value, enabling in this way the definition of Markov Random Fields
         * instead of Conditional Random Fields.
         * \param pot: potentials for the node.
         */
        void setFinalPotentials ( const Eigen::VectorXd &pot )
        {
            assert( pot.rows() == m_type->getNumberOfClasses() );

            m_potentials            = pot;
            m_handCodedPotentials   = true;
        }

        /** Returns if the potentials of the node were set manually and then are final.
         *  \return True if the potentials of the node were set manually, false
         *           otherwise.
         */
        inline bool finalPotentials(){ return m_handCodedPotentials; }

        /** Function for setting the multipliers for the node classes.
         * \param newMultipliers: new multipliers to be set.
         */
        inline void setClassMultipliers( Eigen::VectorXd & newMultipliers)
        {
            assert( newMultipliers.size() == m_type->getNumberOfClasses() );
            m_classMultipliers = newMultipliers;
        }

        template <typename T> void setClassMultipliers( T &multipliers )
        {
            Eigen::VectorXd eigen_multipliers = vectorToEigenVector( multipliers );
            setClassMultipliers( eigen_multipliers );
        }

        /** This method returns a reference to the vector of class multipliers.
         *  \return Reference to the vector of class multipliers.
         */
        inline Eigen::VectorXd & getClassMultipliers()
        {
            return m_classMultipliers;
        }

        /** This method returns a copy of the vector of class multipliers.
         *  \return Copy of the vector of class multipliers.
         */
        inline Eigen::VectorXd getClassMultipliers() const
        {
            return m_classMultipliers;
        }

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
            ar << m_fixed;
            ar << m_handCodedPotentials;
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_id;
            ar & m_type;
            ar & m_label;
            ar >> m_features;
            ar >> m_potentials;
            ar >> m_fixed;
            ar >> m_handCodedPotentials;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CNode, 1)

#endif
