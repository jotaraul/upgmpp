
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

#ifndef _UPGMpp_INFERENCE_MAP_
#define _UPGMpp_INFERENCE_MAP_

#include "base.hpp"
#include "inference_utils.hpp"

namespace UPGMpp
{
    class CInferenceMAP
    {
    protected:
        TInferenceOptions                       m_options;
        std::map<size_t,std::vector<size_t> >   m_mask;
        double                                  m_executionTime; // execution time of the last inference in ns

    public:

        virtual void infer( CGraph &graph, std::map<size_t,size_t> &results, bool debug=false ) = 0;

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

        /** Returns the execution time of the last time that the inference process
          * was launch (in seconds).
          */
        inline double getExecutionTime() const { return m_executionTime*pow(10,-9); }

    };

    class CMaxNodePotInferenceMAP : public CInferenceMAP
    {
    public:
        void infer( CGraph &graph, std::map<size_t,size_t> &results, bool debug=false );
    };

    class CICMInferenceMAP : public CInferenceMAP
    {
    public:
        void infer( CGraph &graph, std::map<size_t,size_t> &results, bool debug=false );
    };

    class CICMGreedyInferenceMAP : public CInferenceMAP
    {
    public:
        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CExactInferenceMAP : public CInferenceMAP
    {
    public:
        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CLBPInferenceMAP : public CInferenceMAP
    {
    public:
        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CTRPBPInferenceMAP : public CInferenceMAP
    {
    public:
        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CRBPInferenceMAP : public CInferenceMAP
    {
    public:
        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CGraphCutsInferenceMAP : public CInferenceMAP
    {
    public:
        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CAlphaExpansionInferenceMAP : public CInferenceMAP
    {
    public:

        CAlphaExpansionInferenceMAP()
        {
            // How to face supermodular potentials/energies. Options:
            // "QPBO"
            // "truncate"
            // "ignore"
            m_options.particularS["submodularApproach"] = "QPBO";
        }

        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CAlphaBetaSwapInferenceMAP : public CInferenceMAP
    {
    public:

        CAlphaBetaSwapInferenceMAP()
        {
            // How to face supermodular potentials/energies. Options:
            // "QPBO"
            // "truncate"
            // "ignore"
            m_options.particularS["supermodularApproach"] = "originalQPBO";
        }

        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

    class CRestartsInferenceMAP : public CInferenceMAP
    {
    public:

        CRestartsInferenceMAP()
        {
            m_options.particularS["method"] = "ICM";
        }

        void infer(CGraph &graph, std::map<size_t, size_t> &results, bool debug=false);
    };

}

#endif
