
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

#ifndef _UPGMpp_EDGE_TYPE_
#define _UPGMpp_EDGE_TYPE_

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
    public:

        /** Default constructor
         */
        CEdgeType()
        {
            static size_t ID = 0;
            m_ID = ID;
            ID++;
        }

        /** Additional constructor
         */
        CEdgeType( size_t N_features, size_t N_classes1, size_t N_classes2 )
        {
            static size_t ID = 0;
            m_ID = ID;
            ID++;

            m_weights.resize( N_features );

            for ( size_t i = 0; i < N_features; i++ )
                m_weights[i].resize( N_classes1, N_classes2 );
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

        /** Establishs a new vector of matrices of weights.
         *  \param weights: new vector of matrices of weights, where each vector
         *                  position refers to an edge feature.
         */
        inline void setWeights( std::vector<Eigen::MatrixXd> &weights ) { m_weights = weights; }

        inline void setWeight( size_t feature, size_t row, size_t col, double &weight)
        {
            m_weights[feature](row,col) = weight;
        }

        /** Returns the vector of matrices of weights.
         * \return A reference to the vector.
         */
        inline std::vector<Eigen::MatrixXd>& getWeights() { return m_weights; }

        /**	Function for prompting the content of a edge type
          * \return An stream with the edge type information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CEdgeType& n)
        {
            output << "ID: " << n.getID() << std::endl;

            return output;
        }
    };

}

#endif //_UPGMpp_EDGE_TYPE_
