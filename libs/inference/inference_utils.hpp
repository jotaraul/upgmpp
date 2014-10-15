
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
        string initialAssignation; // Random, MaxNodePotential

        // Particular options for the different methods
        map<std::string,double>        particularD;
        map<std::string,bool>          particularB;
        map<std::string,std::string>   particularS;

        TInferenceOptions() : maxIterations( 100 ),
                              convergency( 0.0001),
                              considerNodeFixedValues ( false ),
                              initialAssignation( "MaxNodePotential")
        {}
    };

    extern size_t messagesLBP( CGraph &graph,
                               TInferenceOptions &options,
                               std::vector<std::vector<Eigen::VectorXd> > &messages,
                               bool maximize = true,
                               const vector<size_t> &tree = vector<size_t>());

    extern void getSpanningTree( CGraph &graph, std::vector<size_t> &tree);


    extern int fordFulkerson(MatrixXd &graph, int s, int t, VectorXi &cut);

    void getMostProbableNodeAssignation( CGraph &graph,
                                         std::map<size_t,size_t> &assignation,
                                         TInferenceOptions &options);

    void getRandomAssignation(CGraph &graph,
                              std::map<size_t,size_t> &assignation,
                              TInferenceOptions &options);

    void applyMaskToPotentials(CGraph &graph, map<size_t,vector<size_t> > &mask );
}

#endif
