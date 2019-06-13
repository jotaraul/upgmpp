
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

#ifndef _UPGMpp_EDGE_TYPE_
#define _UPGMpp_EDGE_TYPE_

#include "CNodeType.hpp"
#include "base_utils.hpp"

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <iostream>
#include <boost/shared_ptr.hpp>

namespace UPGMpp
{
    /** Anticipated class declararions */

    class CEdgeType;

    /** Useful type definitions */

    typedef boost::shared_ptr<CEdgeType> CEdgeTypePtr;

    /** Function used to compute the potentials of an edge of this type. It can
     * be set by the user ;).
     * \param weights: vector of matrices of weights trained for this edge type.
     * \param features: Features of the edge.
     * \return The edge potentials.
     */
    extern Eigen::MatrixXd linearModelEdge(std::vector<Eigen::MatrixXd> &weights, Eigen::VectorXd &features);

    class CEdgeType
    {
        /** Given that multiple types of nodes can coexist in a same graph, the
         *  edges between them will be probably of different types as well. This
         *  class implements the needed functionality for the creation and
         *  explotation of a given edge type.
         */
    private:
        std::vector<Eigen::MatrixXd> m_weights;     //!< Vector of matrices of weights. Each position of a vector correspond with a matrix for a certain edge feature.                
        size_t      m_ID;       //!< ID of the type of Edge. Must be unique (this is transparent for the user).
        std::string m_label;    //!< A label assigned to this type of edge. E.g. "Edge between an object and a place".
        Eigen::MatrixXd (*m_computePotentialsFunction)( std::vector<Eigen::MatrixXd> &, Eigen::VectorXd &);
        CNodeTypePtr m_nodeType1; //!< Node type of the first node.
        CNodeTypePtr m_nodeType2; //!< Node type of the second node.

        size_t                    m_nFeatures;    //!< Number of edge features.
        std::vector< std::string> m_featureNames; //!< Name of the edge features. Not mandatory.

        /** Private function for obtaning the ID of a new edge type.
         */
        size_t setID(){ static size_t ID = 0; return ID++; }

    public:

        /** Default constructor
         */
        CEdgeType(): m_nFeatures(0)
        {
            m_computePotentialsFunction = &linearModelEdge;

            m_ID = setID();
        }

        /** Additional constructor
         */
        CEdgeType( size_t N_features,
                   CNodeTypePtr nodeType1,
                   CNodeTypePtr nodeType2,
                   std::string label="")
        {
            m_computePotentialsFunction = &linearModelEdge;

            // Get unique ID            
            m_ID = setID();

            m_label = label;
            m_nFeatures = N_features;

            size_t N_classes_1;
            size_t N_classes_2;

            // Ensure that the node type with the lower ID is always the first
            // node type in the edge type.
            if ( nodeType1->getID() <= nodeType2->getID() )
            {
                m_nodeType1 = nodeType1;
                m_nodeType2 = nodeType2;

                N_classes_1 = nodeType1->getNumberOfClasses();
                N_classes_2 = nodeType2->getNumberOfClasses();
            }
            else
            {
                m_nodeType1 = nodeType2;
                m_nodeType2 = nodeType1;

                N_classes_1 = nodeType2->getNumberOfClasses();
                N_classes_2 = nodeType1->getNumberOfClasses();
            }

            // Create the vector of matrices of weights, one per edge feature
            m_weights.resize( N_features );

            for ( size_t i = 0; i < N_features; i++ )
            {
                m_weights[i].resize( N_classes_1, N_classes_2 );
                m_weights[i].fill( 0 );
            }
        }

        /** Additional constructor
         */
        CEdgeType( const std::vector< std::string > &featureNames,
                   CNodeTypePtr nodeType1,
                   CNodeTypePtr nodeType2,
                   const std::string label="")
            : m_featureNames( featureNames )

        {
            m_computePotentialsFunction = &linearModelEdge;

            // Get unique ID
            m_ID = setID();

            m_nFeatures = featureNames.size();

            size_t N_classes_1;
            size_t N_classes_2;

            // Ensure that the node type with the lower ID is always the first
            // node type in the edge type.
            if ( nodeType1->getID() <= nodeType2->getID() )
            {
                m_nodeType1 = nodeType1;
                m_nodeType2 = nodeType2;

                N_classes_1 = nodeType1->getNumberOfClasses();
                N_classes_2 = nodeType2->getNumberOfClasses();
            }
            else
            {
                m_nodeType1 = nodeType2;
                m_nodeType2 = nodeType1;

                N_classes_1 = nodeType2->getNumberOfClasses();
                N_classes_2 = nodeType1->getNumberOfClasses();
            }

            // Create the vector of matrices of weights, one per edge feature
            m_weights.resize( m_nFeatures );

            for ( size_t i = 0; i < m_nFeatures; i++ )
            {
                m_weights[i].resize( N_classes_1, N_classes_2 );
                m_weights[i].fill(0);
            }
        }
        /**	Function for retrieving the ID of the edge type.
          * \return A copy of the ID of the edge type.
          */
        inline size_t getID() const { return m_ID; }

        /**	Function for retrieving the ID of the edge type.
          * \return A reference to the ID of the edge type.
          */
        inline size_t& getID() { return m_ID; }


        /**	Set the label for this type of edge.
          * \param label: New label.
          */
        inline void setLabel( std::string &label ) { m_label = label; }

        /**	Function for retrieving the label of the edge type.
          * \return A copy of the label of the edge type.
          */
        inline std::string getLabel() const { return m_label; }

        /**
         */
        inline CNodeTypePtr getN1Type() { return m_nodeType1; }

        /** Establishs a new vector of matrices of weights.
         *  \param weights: new vector of matrices of weights, where each vector
         *                  position refers to an edge feature.
         */
        void setWeights( std::vector<Eigen::MatrixXd> &weights )
        {
            // Check if the default constructor was used, if so the matrix of
            // weights could be empty
            if ( m_nFeatures )
            {
                assert( weights.size() == m_nFeatures );

                for ( size_t i = 0; i < m_nFeatures; i++ )
                {
                    assert( m_weights[i].cols() == weights[i].cols() );
                    assert( m_weights[i].rows() == weights[i].rows() );
                }
            }
            else
                m_nFeatures = weights.size();

            m_weights = weights;
        }

        inline void setWeight( size_t feature, size_t row, size_t col, double &weight)
        {
            assert ( feature < m_nFeatures );
            assert ( row < m_weights[feature].rows() );
            assert ( col < m_weights[feature].cols() );

            m_weights[feature](row,col) = weight;
        }

        /**	Reset weights to 0.
        */
        void resetWeights()
        {
            for ( size_t i = 0; i < m_nFeatures; i++ )
            {
                m_weights[i].resize( m_nodeType1->getNumberOfClasses(),
                                     m_nodeType2->getNumberOfClasses() );
                m_weights[i].fill( 0 );
            }
        }

        /** Returns the vector of matrices of weights.
         * \return A reference to the vector.
         */
        inline std::vector<Eigen::MatrixXd>& getWeights() { return m_weights; }
        std::vector<Eigen::MatrixXd> getWeights() const { return m_weights; }

        /** Returns the number of edge features
          * \return The number of edge features.
          */
        inline size_t getNumberOfFeatures() { return m_nFeatures; }


        /** Set the name of the edge features.
         * \param newNames: names for the edge features.
         */
        void setFeatureNames( const std::vector< std::string > &newNames )
        {
            if ( m_nFeatures )
                assert ( m_nFeatures == newNames.size() );

            m_featureNames = newNames;
        }

        /** Get the name of the edge features.
         * \return A copy of the vector of edge features.
         */
        inline std::vector<std::string> getFeatureNames(){ return m_featureNames; }


        /** Function that computes the potentials of an edge given their features.
         * \param features: features of the edge.
         * \return The edge potentials.
         **/
        Eigen::MatrixXd computePotentials( Eigen::VectorXd &features )
        {
            assert ( features.rows() == m_nFeatures );

            return (*m_computePotentialsFunction)( m_weights, features );
        }

        /** Set the function for computing the edge potentials
         * \param newFunction: New function for computing the potentials.
         */
        void setComputePotentialsFunction(
                Eigen::MatrixXd (*newFunction)
                (std::vector<Eigen::MatrixXd> &, Eigen::VectorXd &) )
        {
            m_computePotentialsFunction = newFunction;
        }

        /**	Function for prompting the content of a edge type
          * \return An stream with the edge type information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CEdgeType& n)
        {
            output << "ID: " << n.getID() << std::endl;
            output << "Label: " << n.getLabel() << std::endl;
            std::vector<Eigen::MatrixXd> weights = n.getWeights();
            output << "Weights: " << std::endl;

            for ( size_t feat = 0; feat < weights.size(); feat++ )
            {
                output << "For feature " << feat << ":" << std::endl;
                output << weights[feat] << endl;
            }

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
            ar & m_weights;
            ar & m_ID;
            ar & m_label;
            ar & m_nodeType1;
            ar & m_nodeType2;

            ar & m_nFeatures;
            ar & m_featureNames;
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar & m_weights;
            ar & m_ID;
            ar & m_label;
            ar & m_nodeType1;
            ar & m_nodeType2;

            ar & m_nFeatures;
            ar & m_featureNames;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

}

BOOST_CLASS_VERSION(UPGMpp::CEdgeType, 1)

#endif //_UPGMpp_EDGE_TYPE_
