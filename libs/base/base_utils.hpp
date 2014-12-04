
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

#ifndef _UPGMpp_BASE_UTILS_
#define _UPGMpp_BASE_UTILS_

#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define DEBUG(m) if(debug) cout << m << endl;

#define DEBUGD(m,d) if(debug) cout << m << d << endl;



namespace boost
{
template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(
    Archive & ar,
    Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t,
    const unsigned int file_version
)
{
    size_t rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    if( (unsigned int)(rows * cols) != t.size() )
    t.resize( rows, cols );

    for(unsigned int i=0; i<t.size(); i++)
    ar & t.data()[i];
}
}

namespace UPGMpp
{

    #define SHOW_VECTOR(s,v)            \
        std::cout<<s;                   \
        for(size_t i=0;i<v.size();i++)  \
        {                               \
            cout<<v[i]<<" ";            \
        }                               \
        std::cout<<std::endl;

    #define SHOW_VECTOR_NODES_ID(s,v)   \
        std::cout<<s;                   \
        for(size_t i=0;i<v.size();i++)  \
        {                               \
            cout<<v[i]->getID()<<" ";   \
        }                               \
        std::cout<<std::endl;

    template<typename T> Eigen::VectorXd vectorToEigenVector( T &array )
    {
        Eigen::VectorXd eigenVector;

        size_t N_elements = array.size();

        eigenVector.resize( N_elements );

        for ( size_t i = 0; i < N_elements; i++ )
            eigenVector( i ) = array[i];

        return eigenVector;
    }

    template<typename T> Eigen::VectorXi vectorToIntEigenVector( T &array )
    {
        Eigen::VectorXi eigenVector;

        size_t N_elements = array.size();

        eigenVector.resize( N_elements );

        for ( size_t i = 0; i < N_elements; i++ )
            eigenVector( i ) = array[i];

        return eigenVector;
    }

    template<typename T> bool compareTwoVectors( const T &vector1, const T &vector2 )
    {
        size_t N_v1 = vector1.size();
        size_t N_v2 = vector2.size();

        if ( N_v1 != N_v2 )
            return false;

        for ( size_t i = 0; i < N_v1; i++ )
        {
            std::string value = vector1[i];

            size_t occurencesIn1  = 0;
            size_t occurencesIn2 = 0;

            for ( size_t j = 0; j < N_v1; j++ )
            {
                if ( vector1[j] == value )
                    occurencesIn1++;
                if ( vector2[j] == value )
                    occurencesIn2++;
            }

            if ( occurencesIn1 != occurencesIn2 )
                return false;
        }

        return true;
    }
}

#endif
