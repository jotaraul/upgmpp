
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

    public:

        /** Default constructor
          */
        CNodeType()
        {
            static size_t ID = 0;
            m_ID = ID;
            ID++;
        }

        /** Adittional constructor
         */
        CNodeType( size_t N_classes, size_t N_features )
        {
            static size_t ID = 0;
            m_ID = ID;
            ID++;

            m_weights.resize( N_classes, N_features );
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

        /** Set the weights for this type of node.
          * \param weight: Matrix of the new type of node weights.
          */
        inline void setWeights( Eigen::MatrixXd &weights ) { m_weights = weights; }

        /**	Get the matrix of weights.
          * \return A copy of the matrix of weights.
          */
        inline Eigen::MatrixXd getWeights() const { return m_weights; }

        /**	Get the matrix of weights.
          * \return A reference to the matrix of weights.
          */
        inline Eigen::MatrixXd& getWeights(){ return m_weights; }

        inline size_t getClasses() { return m_weights.rows(); }


        /**	Function for prompting the content of a node type
          * \return An stream with the node type information dumped into it.
          */
        friend std::ostream& operator<<(std::ostream& output, const CNodeType& n)
        {
            output << "ID: " << n.getID() << std::endl;

            return output;
        }

    };
}

#endif //_UPGMpp_NODE_TYPE_
