
#ifndef _UPGMpp_INFERENCE_UTILS_
#define _UPGMpp_INFERENCE_UTILS_

#include "base.hpp"
#include <vector>

namespace UPGMpp
{
    #define NEIGHBORS_IT pair<multimap<size_t,CEdgePtr>::iterator,multimap<size_t,CEdgePtr>::iterator >

    struct TInferenceOptions
    {
        size_t maxIterations;
        double convergency;
        bool   considerNodeFixedValues;

        TInferenceOptions() : maxIterations( 100 ),
                              convergency( 0.0001),
                              considerNodeFixedValues ( false )
        {}
    };

    extern size_t messagesLBP( CGraph &graph,
                               TInferenceOptions &options,
                               std::vector<std::vector<Eigen::VectorXd> > &messages,
                               bool maximize = true);

    extern int fordFulkerson(MatrixXd &graph, int s, int t, VectorXi &cut);
}

#endif
