/*
 * Lamg.h
 *
 *  Created on: Oct 20, 2015
 *      Author: Michael Wegner (michael.wegner@student.kit.edu)
 */

#ifndef NETWORKIT_CPP_NUMERICS_LAMG_LAMG_H_
#define NETWORKIT_CPP_NUMERICS_LAMG_LAMG_H_

#include <vector>

#include "../LinearSolver.h"
#include "MultiLevelSetup.h"
#include "SolverLamg.h"
#include "../GaussSeidelRelaxation.h"

namespace NetworKit {

/**
 * Represents the interface to the Lean Algebraic Multigrid (LAMG) graph Laplacian linear solver
 * by Oren E. Livne and Achi Brandt.
 * @see Livne, Oren E., and Achi Brandt. "Lean algebraic multigrid (LAMG): Fast graph Laplacian linear solver." SIAM Journal on Scientific Computing 34.4 (2012): B499-B522.
 */
class Lamg : public LinearSolver {
private:
	bool validSetup;
	GaussSeidelRelaxation smoother;
	MultiLevelSetup lamgSetup;
	CSRMatrix laplacianMatrix;
	std::vector<LevelHierarchy> compHierarchies;
	std::vector<SolverLamg> compSolvers;
	std::vector<LAMGSolverStatus> compStati;

	std::vector<Vector> initialVectors;
	std::vector<Vector> rhsVectors;

	count numComponents;
	std::vector<std::vector<index>> components;
	std::vector<index> graph2Components;

	void initializeForOneComponent();

public:
	/**
	 * Construct a solver with the desired residual reduction @a desiredResidualRed.
	 * @param desiredResidualRed
	 */
	Lamg(const double desiredResidualRed = 1e-6);
	/** Default destructor */
	~Lamg() = default;

	/**
	 * Compute the multigrid hierarchy for te given Laplacian matrix @a laplacianMatrix.
	 * @param laplacianMatrix
	 * @note This method also works for disconnected graphs. If you know that the graph is connected,
	 * if is faster to use @ref setupConnected instead.
	 */
	void setup(const CSRMatrix &laplacianMatrix);

	/**
	 * Compute the multigrid hierarchy for te given Laplacian matrix @a laplacianMatrix.
	 * @param laplacianMatrix
	 * @note The graph has to be connected for this method to work. Otherwise the output is undefined.
	 */
	void setupConnected(const CSRMatrix &laplacianMatrix);

	/**
	 * Computes the @a result for the matrix currently setup and the right-hand side @a rhs.
	 * The maximum spent time can be specified by @a maxConvergenceTime and the maximum number of iterations can be set
	 * by @a maxIterations.
	 * @param rhs
	 * @param result
	 * @param maxConvergenceTime
	 * @param maxIterations
	 * @return A @ref SolverStatus object which provides some statistics like the final absolute residual.
	 */
	SolverStatus solve(const Vector &rhs, Vector &result, count maxConvergenceTime = 5 * 60 * 1000, count maxIterations = std::numeric_limits<count>::max());

};

} /* namespace NetworKit */

#endif /* NETWORKIT_CPP_NUMERICS_LAMG_LAMG_H_ */
