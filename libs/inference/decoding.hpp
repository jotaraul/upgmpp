
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

#ifndef _UPGMpp_DECODING_
#define _UPGMpp_DECODING_

#include "base.hpp"
#include "inference_utils.hpp"

namespace UPGMpp
{
    class CMAPDecoder
    {
    protected:
        TInferenceOptions                       m_options;
        std::map<size_t,std::vector<size_t> >   m_mask;

    public:

        virtual void decode( CGraph &graph, std::map<size_t,size_t> &results ) = 0;

        inline void setOptions ( TInferenceOptions &options ) { m_options = options; }

        inline void setMask ( std::map<size_t,std::vector<size_t> > &mask )
            { m_mask = mask; }

    };

    class CDecodeICM : public CMAPDecoder
    {
    public:
        void decode( CGraph &graph, std::map<size_t,size_t> &results );
    };

    class CDecodeICMGreedy : public CMAPDecoder
    {
    public:
        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

    class CDecodeExact : public CMAPDecoder
    {
    public:
        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

    class CDecodeLBP : public CMAPDecoder
    {
    public:
        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

}

#endif
