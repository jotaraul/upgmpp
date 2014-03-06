
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
    if( rows * cols != t.size() )
    t.resize( rows, cols );

    for(size_t i=0; i<t.size(); i++)
    ar & t.data()[i];
}
}

#endif
