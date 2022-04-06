// no-networkit-format
/*
 * APSP.cpp
 *
 *  Created on: 07.07.2015
 *      Author: Arie Slobbe
 */

#include <omp.h>

#include <networkit/distance/APSP.hpp>
#include <networkit/distance/BFS.hpp>
#include <networkit/distance/Dijkstra.hpp>

namespace NetworKit {

APSP::APSP(const Graph &G) : Algorithm(), G(G) {}

void APSP::run() {
    count n = G.numberOfNodes();
    std::vector<edgeweight> distanceVector(n, 0.0);
    distances.resize(n, distanceVector);

    count nThreads = omp_get_max_threads();
    sssps.resize(nThreads);
#pragma omp parallel
    {
        omp_index i = omp_get_thread_num();
        if (G.isWeighted())
            sssps[i] = std::unique_ptr<SSSP>(new Dijkstra(G, 0, false));
        else
            sssps[i] = std::unique_ptr<SSSP>(new BFS(G, 0, false));
    }

    // Create a vector of nodes so we can have random access for parallelization.
    auto nodeRange = G.nodeRange();
    std::vector<node> nodes(nodeRange.begin(), nodeRange.end());
#pragma omp parallel for schedule(dynamic)
    for (omp_index i = 0; i < n; ++i) {
        auto sssp = sssps[omp_get_thread_num()].get();
        sssp->setSource(nodes[i]);
        sssp->run();
        distances[i] = sssp->getDistances();
    }

    hasRun = true;
}

} /* namespace NetworKit */
