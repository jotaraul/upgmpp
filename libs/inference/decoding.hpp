
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

        inline void setOptions ( TInferenceOptions &options )
        {
            m_options.maxIterations = options.maxIterations;
            m_options.considerNodeFixedValues = options.considerNodeFixedValues;
            m_options.convergency = options.convergency;
            m_options.initialAssignation = options.initialAssignation;

            // Instead of assign the different maps to the local options maps,
            // just insert them. This permits to use default parameters in
            // addiction to the user specified ones.

            m_options.particularD.insert(options.particularD.begin(),
                                         options.particularD.end());

            m_options.particularB.insert(options.particularB.begin(),
                                         options.particularB.end());

            m_options.particularS.insert(options.particularS.begin(),
                                         options.particularS.end());
        }

        inline void setMask ( std::map<size_t,std::vector<size_t> > &mask )
            { m_mask = mask; }

    };

    class CDecodeMaxNodePot : public CMAPDecoder
    {
    public:
        void decode( CGraph &graph, std::map<size_t,size_t> &results );
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

    class CDecodeTRPBP : public CMAPDecoder
    {
    public:
        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

    class CDecodeGraphCuts : public CMAPDecoder
    {
    public:
        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

    class CDecodeAlphaExpansion : public CMAPDecoder
    {
    public:

        CDecodeAlphaExpansion()
        {
            // How to face supermodular potentials/energies. Options:
            // "QPBO"
            // "truncate"
            // "ignore"
            m_options.particularS["submodularApproach"] = "QPBO";
        }

        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

    class CDecodeAlphaBetaSwap : public CMAPDecoder
    {
    public:

        CDecodeAlphaBetaSwap()
        {
            // How to face supermodular potentials/energies. Options:
            // "QPBO"
            // "truncate"
            // "ignore"
            m_options.particularS["supermodularApproach"] = "originalQPBO";
        }

        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

    class CDecodeWithRestarts : public CMAPDecoder
    {
    public:

        CDecodeWithRestarts()
        {
            m_options.particularS["method"] = "ICM";
        }

        void decode(CGraph &graph, std::map<size_t, size_t> &results);
    };

}

#endif
