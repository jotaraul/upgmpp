
#ifndef _UPGMpp_DECODING_
#define _UPGMpp_DECODING_

#include "base.hpp"

namespace UPGMpp
{
    struct TOptions
    {
        size_t maxIterations;
    };


    extern size_t decodeICM( CGraph &graph,
                             TOptions &decodingOptions,
                             std::map<size_t,size_t> &results );

    extern size_t decodeICMGreedy( CGraph &graph,
                             TOptions &decodingOptions,
                             std::map<size_t,size_t> &results );

}

#endif
