
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

#ifndef _UPGMpp_INFERENCE_
#define _UPGMpp_INFERENCE_

#include "base.hpp"
#include "inference_utils.hpp"

namespace UPGMpp
{
    class CInference
    {
    protected:

        TInferenceOptions                       m_options;
        std::map<size_t,std::vector<size_t> >   m_mask; //!< This makes sense in inference?

    public:

        virtual void infer(CGraph &graph,
                           std::map<size_t,Eigen::VectorXd> &nodeBeliefs,
                           std::map<size_t,Eigen::MatrixXd> &edgeBeliefs,
                           double &logZ) = 0;

        inline void setOptions ( TInferenceOptions &options ) { m_options = options; }

        inline void setMask ( std::map<size_t,std::vector<size_t> > &mask )
            { m_mask = mask; }

    };

    class CLBPInference : public CInference
    {
    public:
        void infer(CGraph &graph,
                   std::map<size_t,Eigen::VectorXd> &nodeBeliefs,
                   std::map<size_t,Eigen::MatrixXd> &edgeBeliefs,
                   double &logZ);
    };

    class CTRPBPInference : public CInference
    {
    public:
        void infer(CGraph &graph,
                   std::map<size_t,Eigen::VectorXd> &nodeBeliefs,
                   std::map<size_t,Eigen::MatrixXd> &edgeBeliefs,
                   double &logZ);
    };

    class CRBPInference : public CInference
    {
    public:
        void infer(CGraph &graph,
                   std::map<size_t,Eigen::VectorXd> &nodeBeliefs,
                   std::map<size_t,Eigen::MatrixXd> &edgeBeliefs,
                   double &logZ);
    };
}

#endif
