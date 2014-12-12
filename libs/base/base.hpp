
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

#ifndef _UPGMpp_BASE_
#define _UPGMpp_BASE_

#include "CNodeType.hpp"
#include "CEdgeType.hpp"
#include "CNode.hpp"
#include "CEdge.hpp"
#include "CGraph.hpp"
#include "base_utils.hpp"

#define UPGMpp_TRY		try {

#define UPGMpp_CATCH			\
	}				\
	catch(std::exception &e)	\
	{				\
		cout << "Error: " << e.what();	\
	}	

#endif
