
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

#ifndef _UPGMpp_NODE_TYPE_
#define _UPGMpp_NODE_TYPE_

#include "base_utils.hpp"

#include <string>
#include <Eigen/Dense>

#include <iostream>
#include <boost/shared_ptr.hpp>


namespace UPGMpp
{
    /** Needed anticipated declaration */
    class CNodeType;

    /** Useful type definitions */
    typedef boost::shared_ptr<CNodeType> CNodeTypePtr;

    /** Function used to compute the potentials of a node of this type. It can
     * be set by the user ;). This does:
     * nodePotential = exp(weight_1*feature_1),..,exp(weight_n*feature_n).
     * \param weights: Matrix of weights trained for this node type.
     * \param features: Features of the node.
     * \return The node potentials.
     */
    extern Eigen::VectorXd linearModelNode( Eigen::MatrixXd &weights, Eigen::VectorXd &features );

    class CNodeType
    {
        /**
          * This class enables an interesting feature of PGMs: nodes can be of
          * different types. e.g., a node can represent an object, and other one
          * in the same graph the functionality of a certain location.
          **/
    private:
        Eigen::MatrixXd m_weights;     //!< Matrix of weights. Rows are classes and columns are features
        size_t          m_ID;          //!< ID of the NodeType. Must be unique (transparent to the user).
        std::string     m_label;       //!< Human readable label. Not mandatory.
        Eigen::VectorXd (*m_computePotentialsFunction)( Eigen::MatrixXd &, Eigen::VectorXd &); //!< Pointer to the function that computes the node potentials

        std::vector< std::string >  m_classNames; //!< A vector containing the name of the classes of the node type
        std::vector< std::string >  m_featureNames;//!< Names of the features used to characterize this node type
        size_t  m_nClasses;
        size_t  m_nFeatures;

        /** Private function for obtaning the ID of a new node type.
         */
        size_t setID() const { static size_t ID = 0; return ID++; }

    public:

        /** Default constructor
          */
        CNodeType() : m_nClasses(0)
        {
            m_computePotentialsFunction = &linearModelNode;

            m_ID = setID();
        }

        /** Adittional constructor
         */
        CNodeType( size_t N_classes, size_t N_features, std::string label = "" )
        {
            m_computePotentialsFunction = &linearModelNode;

            m_ID = setID();

            m_nClasses  = N_classes;
            m_nFeatures = N_features;
            m_weights.resize( N_classes, N_features );
            m_label     = label;

            m_classNames.resize( N_classes, "" );
            m_featureNames.resize( N_features, "" );
        }

        /** Adittional constructor
         */
        CNodeType( const std::vector< std::string> &classNames,
                   const std::vector< std::string> &featureNames,
                   const std::string label = "" )
            : m_classNames( classNames ),
              m_featureNames( featureNames )
        {
            m_computePotentialsFunction = &linearModelNode;

            m_ID = setID();

            m_nClasses  = m_classNames.size();
            m_nFeatures = m_featureNames.size();
            m_weights.resize( m_nClasses, m_nFeatures );
            m_label     = label;

        }

        /**	Function for retrieving the ID of the node type.
          * \return The ID of the node type.
          */
        inline size_t getID() const { return m_ID; }

        /**	Function for retrieving the ID of the node type.
          * \return Reference to the ID of the node type.
          */
        inline size_t& getID() { return m_ID; }


        /**	Changes the label of the node type.
          * \param label: label of the node type.
          */
        inline void setLabel( std::string &label ) { m_label = label; }

        /**	Function for retrieving the label of the node type.
          * \return Reference to the label of the node type.
          */
        inline std::string& getLabel(){ return m_label; }

        /**	Function for retrieving the label of the node type.
          * \return Copy of the label of the node type.
          */
        inline std::string getLabel() const { return m_label; }


        /** Set the weights for this type of node.
          * \param weight: Matrix of the new type of node weights.
          */
        void setWeights( Eigen::MatrixXd &weights )
        {
            if ( m_nClasses ) // Node type already initialized
            {
                assert( weights.rows() == m_nClasses );
                assert( weights.cols() == m_nFeatures );

                m_weights = weights;
            }
            else // The default constructor was called, so initialize some
                // needed vbles here
            {
                m_nClasses = weights.rows();
                m_nFeatures = weights.cols();

                // Check if the class names have already been set
                if ( m_classNames.size() )
                    assert( m_nClasses = m_classNames.size() );
                else
                    m_classNames.resize( m_nClasses, "" );

                // Check if the feature names have already been set
                if ( m_featureNames.size() )
                    assert( m_nFeatures = m_featureNames.size() );
                else
                    m_featureNames.resize( m_nFeatures, "" );

                m_weights = weights;
            }

        }

        /**	Get the matrix of weights.
          * \return A copy of the matrix of weights.
          */
        inline Eigen::MatrixXd getWeights() const { return m_weights; }

        /**	Get the matrix of weights.
          * \return A reference to the matrix of weights.
          */
        inline Eigen::MatrixXd& getWeights(){ return m_weights; }


        /** Get the number of classes (states) of nodes of this type.
         * \return Number of classes.
         */
        inline size_t getNumberOfClasses() { return m_nClasses; }


        /** Set the names of the classes of the node type. This method fails if
         * the number of node classes has been already set and the size of the
         * newClassNames vector is different.
         * \param newClassNames: vector of new classe names.
         */
        inline void setClassNames( const std::vector< std::string > &newClassNames )
        {
            if ( m_nClasses )
                assert ( m_nClasses == newClassNames.size() );

            m_classNames = newClassNames;
        }

        /** Get the name of the class used in this node type.
         *  \return A copy of the vector of class names.
         */
        inline std::vector< std::string > getClassNames(){ return m_classNames; }


        /** Set the names of the features of the node type. This method fails if
         * the number of node features has been already set and the size of the
         * newFeaturenames vector is different.
         * \param newFeaturenames: vector of new feature names.
         */
        void setFeatureNames( const std::vector< std::string > &newFeatureNames )
        {
            if ( m_nFeatures )
                assert ( m_nFeatures == newFeatureNames.size() );

            m_featureNames = newFeatureNames;
        }

        /** Get the name of the features used in this node type.
         *  \return A copy of the vector of feature names.
         */
        inline std::vector< std::string > getFeatureNames(){ return m_featureNames; }

        /** Get the number of features of nodes of this type.
         * \return Number of features.
         */
        inline size_t getNumberOfFeatures() { return m_nFeatures; }


        /**	Function for prompting the content of a node type
          * \return An stream with the node type information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CNodeType& n)
        {
            output << "ID: " << n.getID() << std::endl;
            output << "Label: " << n.getLabel() << std::endl;
            output << "Weights: " << std::endl << n.getWeights() << std::endl;

            return output;
        }

        /** Compute the potential of a node according to the weights and its
         * features.
         * \param features: features of the node.
         * \return potentials of that node.
         */
        Eigen::VectorXd computePotentials( Eigen::VectorXd &features )
        {
            // Check that features are a column vector, and transpose it  otherwise
            if ( features.cols() > 1 )
                        features = features.transpose();

            assert ( features.rows() == m_nFeatures );

            return (*m_computePotentialsFunction)(m_weights,features);
        }

        /** Set the function for computing the potentials
         * \param newFunction: the new function for computing the potentials.
         */
        void setComputePotentialsFunction ( Eigen::VectorXd (*newFunction)( Eigen::MatrixXd &, Eigen::VectorXd &) )
        {
            m_computePotentialsFunction = newFunction;
        }

        //
        //  Serialization stuff
        //
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            // note, version is always the latest when saving            
            ar & m_weights;
            ar & m_ID;
            ar & m_label;

            ar & m_classNames;
            ar & m_featureNames;
            ar & m_nClasses;
            ar & m_nFeatures;

        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_weights;
            ar & m_ID;
            ar & m_label;

            ar & m_classNames;
            ar & m_featureNames;
            ar & m_nClasses;
            ar & m_nFeatures;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CNodeType, 1)

#endif //_UPGMpp_NODE_TYPE_
