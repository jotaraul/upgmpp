
#ifndef _UPGMpp_INFERENCE_UTILS_
#define _UPGMpp_INFERENCE_UTILS_

#include "base.hpp"
#include <vector>

namespace UPGMpp
{

    struct TInferenceOptions
    {
        size_t maxIterations;
        double convergency;
    };

    extern size_t messagesLBP( CGraph &graph,
                               TInferenceOptions &options,
                               std::vector<std::vector<Eigen::VectorXd> > &messages );
}

#endif
