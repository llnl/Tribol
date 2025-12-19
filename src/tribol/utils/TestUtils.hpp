// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_UTILS_TESTUTILS_HPP_
#define SRC_TRIBOL_UTILS_TESTUTILS_HPP_

#include "tribol/common/Parameters.hpp"

// MFEM includes
#include "mfem.hpp"

#include "axom/config.hpp"

namespace tribol {

/// RAII struct to initialize and finalize MPI for tests and examples
struct SimpleMPIWrapper {
  SimpleMPIWrapper( int argc, char* argv[] )
  {
#ifdef TRIBOL_USE_MPI
    MPI_Init( &argc, &argv );
#else
    static_cast<void>( argc );  // elide warning about unused vars
    static_cast<void>( argv );
#endif
  }

  ~SimpleMPIWrapper()
  {
#ifdef TRIBOL_USE_MPI
    MPI_Finalize();
#endif
  }
};

/*!
 * \brief Struct to hold control parameters for tests
 */
struct TestControlParameters {
  /// Constructor
  TestControlParameters()
      : penalty_ratio( false ),
        constant_rate_penalty( false ),
        percent_rate_penalty( false ),
        rate_penalty( 1.0 ),
        rate_penalty_ratio( 0.0 ),
        const_penalty( 1.0 ),
        enable_timestep_vote( false ),
        timestep_pen_frac( 0.30 ),
        timestep_scale( 1.0 )
  {
  }

  ~TestControlParameters()
  {
    // no-op
  }

  RealT dt{ 0. };
  RealT auto_contact_pen_frac{ 0.95 };

  // penalty control parameters
  bool penalty_ratio;
  bool constant_rate_penalty;
  bool percent_rate_penalty;
  RealT rate_penalty;
  RealT rate_penalty_ratio;
  RealT const_penalty;
  bool enable_timestep_vote;
  RealT timestep_pen_frac;
  RealT timestep_scale;
};

/*!
 * \brief class to hold test mesh data
 */
class TestMesh {
 public:
  /// constructor
  TestMesh();

  /// destructor
  ~TestMesh();

  /// clear function
  void clear( bool keepCoords = false  ///< option to keep nodal coordinate arrays for hex-to-tet mesh
  );

  /// performs tribol registration calls and calls tribol::update()
  int tribolSetupAndUpdate( ContactMethod method,           ///< contact method
                            EnforcementMethod enforcement,  ///< constraint enforcement method
                            ContactModel model,             ///< contact model
                            ContactCase contact_case,       ///< contact case
                            bool visualization,             ///< true if visualization
                            TestControlParameters& params   ///< control parameters struct
  );

  /// performs tribol registration calls and calls tribol::update() using "simple" API
  int simpleTribolSetupAndUpdate( ContactMethod method,           ///< contact method
                                  EnforcementMethod enforcement,  ///< constraint enforcement method
                                  ContactModel model,             ///< contact model
                                  ContactCase contact_case,       ///< contact case
                                  bool visualization,             ///< true if visualization
                                  TestControlParameters& params   ///< control parameters struct
  );

  /*!
   * \brief setups of a 3D contact hex mesh consisting of two blocks
   *
   * \param [in] numElemsX1 number of elements in the x-direction for first block
   * \param [in] numElemsY1 number of elements in the y-direction for first block
   * \param [in] numElemsZ1 number of elements in the z-direction for first block
   * \param [in] xMin1 minimum x-coordinate location of first block
   * \param [in] yMin1 minimum y-coordinate location of first block
   * \param [in] zMin1 minimum z-coordinate location of first block
   * \param [in] xMax1 maximum x-coordinate location of first block
   * \param [in] yMax1 maximum y-coordinate location of first block
   * \param [in] zMax1 maximum z-coordinate location of first block
   * \param [in] numElemsX2 number of elements in the x-direction for second block
   * \param [in] numElemsY2 number of elements in the y-direction for second block
   * \param [in] numElemsZ2 number of elements in the z-direction for second block
   * \param [in] xMin2 minimum x-coordinate location of second block
   * \param [in] yMin2 minimum y-coordinate location of second block
   * \param [in] zMin2 minimum z-coordinate location of second block
   * \param [in] xMax2 maximum x-coordinate location of second block
   * \param [in] yMax2 maximum y-coordinate location of second block
   * \param [in] zMax2 maximum z-coordinate location of second block
   * \param [in] thetaMortar angle of rotation of non-z-plane vertices about z-axis
   * \param [in] thetaNonmortar angle of rotation of non-z-plane vertices about z-axis
   *
   */
  void setupContactMeshHex( int numElemsX1, int numElemsY1, int numElemsZ1, RealT xMin1, RealT yMin1, RealT zMin1,
                            RealT xMax1, RealT yMax1, RealT zMax1, int numElemsX2, int numElemsY2, int numElemsZ2,
                            RealT xMin2, RealT yMin2, RealT zMin2, RealT xMax2, RealT yMax2, RealT zMax2,
                            RealT thetaMortar, RealT thetaNonmortar );

  /*!
   * \brief setups of a 3D contact tet mesh consisting of two blocks
   *
   * \param [in] numElemsX1 number of elements in the x-direction for first block
   * \param [in] numElemsY1 number of elements in the y-direction for first block
   * \param [in] numElemsZ1 number of elements in the z-direction for first block
   * \param [in] xMin1 minimum x-coordinate location of first block
   * \param [in] yMin1 minimum y-coordinate location of first block
   * \param [in] zMin1 minimum z-coordinate location of first block
   * \param [in] xMax1 maximum x-coordinate location of first block
   * \param [in] yMax1 maximum y-coordinate location of first block
   * \param [in] zMax1 maximum z-coordinate location of first block
   * \param [in] numElemsX2 number of elements in the x-direction for second block
   * \param [in] numElemsY2 number of elements in the y-direction for second block
   * \param [in] numElemsZ2 number of elements in the z-direction for second block
   * \param [in] xMin2 minimum x-coordinate location of second block
   * \param [in] yMin2 minimum y-coordinate location of second block
   * \param [in] zMin2 minimum z-coordinate location of second block
   * \param [in] xMax2 maximum x-coordinate location of second block
   * \param [in] yMax2 maximum y-coordinate location of second block
   * \param [in] zMax2 maximum z-coordinate location of second block
   * \param [in] thetaMortar angle of rotation of non-z-plane vertices about z-axis
   * \param [in] thetaNonmortar angle of rotation of non-z-plane vertices about z-axis
   *
   */
  void setupContactMeshTet( int numElemsX1, int numElemsY1, int numElemsZ1, RealT xMin1, RealT yMin1, RealT zMin1,
                            RealT xMax1, RealT yMax1, RealT zMax1, int numElemsX2, int numElemsY2, int numElemsZ2,
                            RealT xMin2, RealT yMin2, RealT zMin2, RealT xMax2, RealT yMax2, RealT zMax2,
                            RealT thetaMortar, RealT thetaNonmortar );

  /*!
   * \brief rotates the coordinates of one of the mesh blocks
   *
   * \param [in] mesh_id the id of the block to be rotated
   * \param [in] theta_x rotation in degrees about the x-axis
   * \param [in] theta_y rotation in degrees about the y-axis
   * \param [in] theta_z rotation in degrees about the z-axis
   * \param [in] shift_x x-direction shift for point of rotation
   * \param [in] shift_y y-direction shift for point of rotation
   * \param [in] shift_z z-direction shift for point of rotation
   *
   * \pre the contact mesh must be setup prior to this call
   *
   * \note mortar is mesh id 0, and nonmortar is mesh id 1. Default rotation is
   *       about (0,0,0)
   */
  void rotateContactMesh( const int mesh_id, RealT theta_x = 0., RealT theta_y = 0., RealT theta_z = 0.,
                          RealT shift_x = 0., RealT shift_y = 0., RealT shift_z = 0. );

  /*!
   * \brief translates the coordinates of one of the mesh blocks
   *
   * \param [in] mesh_id the id of the block to be rotated
   * \param [in] shift_x rotation in degrees about the x-axis
   * \param [in] shift_y rotation in degrees about the y-axis
   * \param [in] shift_z rotation in degrees about the z-axis
   *
   * \pre the contact mesh must be setup prior to this call
   *
   * \note mortar is mesh id 0, and nonmortar is mesh id 1
   */
  void translateContactMesh( const int mesh_id, RealT shift_x = 0., RealT shift_y = 0., RealT shift_z = 0. );

  /*!
   * \brief sets up an mfem mesh object representation of the original hex or tet test mesh
   *
   *
   * \param [in] fix_orientation true will fix any orientation errors in original TestMesh
   *
   * \note must call setupContactMeshHex() or setupContactMeshTet() to construct the test mesh
   * prior to calling this routine
   *
   */
  void setupMfemMesh( bool fix_orientation = true );

  /*!
   * \brief sets up the Dirichlet BC node ids and values for a single 3D mesh block
   *        for the contact PATCH TEST
   *
   * \param [in] meshId mesh id for the given block
   * \param [in] numElemsX number of elements in the x-direction
   * \param [in] numElemsY number of elements in the y-direction
   * \param [in] numElemsZ number of elements in the z-direction
   * \param [in] nodeIdOffset node id offset for this block
   * \param [in] inHomogeneousGap true if enforcing gap closure through Dirichlet BCs
   * \param [in] inHomogeneousZVal z-component inhomogeneous Dirichlet BC for when inHomogeneousGap is true
   *
   * \note inHomogeneousGap is used for enforcing gap closure with Dirichlet BCs  instead of
   *       contact enforcement. This is used in tribol/tests/tribol_mortar_pressure_sol.cpp
   *
   */
  void setupPatchTestDirichletBCs( IndexT mesh_id, int numElemsX, int numElemsY, int numElemsZ, int nodeIdOffset,
                                   bool inHomogeneousGap, RealT inHomogeneousZVal = 0. );

  /*!
   * \brief sets up pressure dof ids for a 3D nonmortar mesh block for PATCH TEST
   *
   * \param [in] meshId id of the mesh block for patch test
   * \param [in] numElemsX number of elements in x-direction of nonmortar block
   * \param [in] numElemsY number of elements in y-direction of nonmortar block
   * \param [in] numElemsZ number of elements in z-direction of nonmortar block
   * \param [in] nodeIdOffset nonmortar node id offset
   * \param [in] contact true if enforcing zero gap using contact enforcement
   *
   */
  void setupPatchTestPressureDofs( IndexT mesh_id, int numElemsX, int numElemsY, int numElemsZ, int nodeIdOffset,
                                   bool contact );

  /*!
   * \brief allocates and sets velocity arrays
   *
   * \param [in] valX x-velocity value to set velocity arrays
   * \param [in] valY y-velocity value to set velocity arrays
   * \param [in] valZ z-velocity value to set velocity arrays
   *
   * \note in the future we should handle an array of velocity values
   *
   */
  void allocateAndSetVelocities( IndexT mesh_id, RealT valX, RealT valY, RealT valZ = 0. );

  /*!
   * \brief allocates and sets bulk modulus arrays on mesh
   *
   * \param [in] val single scalar bulk modulus
   *
   * \note in the future we should handle an array of values
   *
   */
  void allocateAndSetBulkModulus( IndexT mesh_id, RealT val );

  /*!
   * \brief allocates and sets element thickness arrays on mesh
   *
   * \param [in] val single scalar element thickness
   *
   * \note this is suitable for a uniform mesh. In the future we
   *       should handle an array of values
   *
   */
  void allocateAndSetElementThickness( IndexT mesh_id, RealT t );

  /*!
   * \brief wraps element Jacobian calculations for linear elasticity
   *
   * \param [in,out] A      mfem sparse matrix data object where Jac. contributions are summed into
   * \param [in]     nu     Poisson's ratio for linear elastic material
   * \param [in]     youngs Young's modulus for linear elastic material
   *
   *
   * \pre TestMesh::mfem_mesh != nullptr
   * \pre matrix A cannot have been finalized yet
   */
  void computeEquilibriumJacobian( mfem::SparseMatrix* const A, RealT const nu, RealT const youngs );

  /*!
   * \brief wraps element Jacobian calculations for linear elasticity
   *
   * \param [in,out] A      mfem sparse matrix data object where Jac. contributions are summed into
   * \param [in]     eInteg mfem elasticity integrator
   * \param [in]     fes    finite element space
   *
   *
   * \pre fes != nullptr
   * \pre TestMesh::mfem_mesh != nullptr
   * \pre matrix A cannot have been finalized yet
   */
  void computeEquilibriumJacobian( mfem::SparseMatrix* const A, mfem::ElasticityIntegrator* eInteg,
                                   mfem::FiniteElementSpace* fes );

  /*!
   * \brief computes all element Jacobian contributions for linear elasticity using MFEM
   *
   * \param [in,out] A      mfem sparse matrix data object where Jac. contributions are summed into
   * \param [in]     eInt   mfem elasticity integrator
   * \param [in]     fes    finite element space
   *
   *
   * \pre fes != nullptr
   * \pre TestMesh::mfem_mesh != nullptr
   * \pre matrix A cannot have been finalized yet
   */
  void computeElementJacobianContributions( mfem::SparseMatrix* const A, mfem::ElasticityIntegrator* eInt,
                                            mfem::FiniteElementSpace* fe_space, bool matrixDebug = false );

  /*!
   * \brief takes an oversized tribol sparse matrix and condenses it to an appropriate
   *        system size sparse matrix
   *
   * \param [in,out] ATribol Tribol-sized dense matrix (dim * totalNumberOfNodes + totalNumberOfNodes)
   * \param [in,out] ASystem system-sized sparse matrix (dim * totalNumberOfNodes + numNonmortarNodes)
   *
   * \pre ATribol != nullptr
   * \pre ASystem != nullptr
   *
   * \note this routine is a little odd taking a dense matrix and populating a sparse
   *       matrix. This reflects the way the Lagrange multiplier patch test code
   *       was originally written and does not represent the optimal way of doing things
   */
  void tribolMatrixToSystemMatrix( mfem::DenseMatrix* const ATribol, mfem::SparseMatrix* const ASystem );

  /*!
   * \brief populate a right hand side length vector with gap evaluations
   *
   * \param [in,out] v pointer to right hand side vector
   *
   * \pre v of length, dim * numTotalNodes + numPressureDofs
   *
   */
  void getGapEvals( RealT* const v );

  /*!
   * \brief Modifies matrix A and rhs vector b to enforce Dirichlet BCs
   *
   * \param [in,out] A mfem sparse matrix
   * \param [in,out] b mfem vector (rhs vector)
   * \param [in]     contact applies BCs different for a contact vs. non-contact solution
   *
   * \note The boolean, contact, is something utilized for the SINGLE_MORTAR LM
   *       patch unit test
   *
   * \pre b of length, dim * numTotalNodes + numPressureDofs
   * \pre A of dimensions (length b) x (length b)
   *
   */
  void enforceDirichletBCs( mfem::SparseMatrix* const A, mfem::Vector* const b, bool contact = true );

  /// print mesh to vtk file
  void testMeshToVtk( const std::string& dir,  ///< Name of the output directory
                      int cycle,               ///< Cycle number
                      RealT time               ///< Simulation time
  );

 public:
  mfem::Mesh* mfem_mesh;

  // Basic info about the mesh
  IndexT mortarMeshId;           ///< Mesh id for mortar portion of mesh
  IndexT nonmortarMeshId;        ///< Mesh id for nonmortar portion of mesh
  int cellType;                  ///< Type of contact surface cell in mesh
  int numTotalNodes;             ///< Total number of nodes in the mesh
  int numMortarNodes;            ///< Number of mortar nodes (not just surface nodes)
  int numNonmortarNodes;         ///< Number of nonmortar nodes (not just surface nodes)
  int numNonmortarSurfaceNodes;  ///< Number of surface nodes on the nonmortar side
  int numTotalElements;          ///< Total number of elements
  int numMortarElements;         ///< Number of mortar elements
  int numNonmortarElements;      ///< Number of nonmortar nodes
  int numTotalFaces;             ///< Total number of faces
  int numMortarFaces;            ///< Number of mortar faces
  int numNonmortarFaces;         ///< Number of nonmortar faces
  int numNodesPerFace;           ///< Number of nodes per face
  int numNodesPerElement;        ///< Number of nodes per element
  int dim;                       ///< Mesh dimension

  // Pointers to Boundary Condition Data
  int* dirNodesX1;   ///< Pointer to list of mortar node ids with x-component Dirichlet BCs
  int* dirNodesY1;   ///< Pointer to list of mortar node ids with y-component Dirichlet BCs
  int* dirNodesZ1;   ///< Pointer to list of mortar node ids with z-component Dirichlet BCs
  RealT* iDirValX1;  ///< Pointer to x-component Dirichlet BC values for specified mortar nodes
  RealT* iDirValY1;  ///< Pointer to y-component Dirichlet BC values for specified mortar nodes
  RealT* iDirValZ1;  ///< Pointer to z-component Dirichlet BC values for specified mortar nodes
  int* dirNodesX2;   ///< Pointer to list of nonmortar node ids with x-component Dirichlet BCs
  int* dirNodesY2;   ///< Pointer to list of nonmortar node ids with y-component Dirichlet BCs
  int* dirNodesZ2;   ///< Pointer to list of nonmortar node ids with z-component Dirichlet BCs
  RealT* iDirValX2;  ///< Pointer to x-component Dirichlet BC values for specified nonmortar nodes
  RealT* iDirValY2;  ///< Pointer to y-component Dirichlet BC values for specified nonmortar nodes
  RealT* iDirValZ2;  ///< Pointer to z-component Dirichlet BC values for specified nonmortar nodes
  int* presDofs1;    ///< Pointer to mortar node ids with a pressure BC
  int* presDofs2;    ///< Pointer to nonmortar node ids with a pressure BC

  // Pointers to connectivity data
  IndexT* faceConn1;  ///< Pointer to mortar face connectivity
  IndexT* faceConn2;  ///< Pointer to nonmortar face connectivity
  IndexT* elConn1;    ///< Pointer to mortar element connectivity
  IndexT* elConn2;    ///< Pointer to nonmortar element connectivity

  // TODO can we make these mfem grid functions?
  RealT* fx1;  ///< Mortar nodal forces, x-component
  RealT* fy1;  ///< Mortar nodal forces, y-component
  RealT* fz1;  ///< Mortar nodal forces, z-component
  RealT* fx2;  ///< Nonmortar nodal forces, x-component
  RealT* fy2;  ///< Nonmortar nodal forces, y-component
  RealT* fz2;  ///< Nonmortar nodal forces, z-component
  RealT* vx1;  ///< Mortar nodal velocities, x-component
  RealT* vy1;  ///< Mortar nodal velocities, y-component
  RealT* vz1;  ///< Mortar nodal velocities, z-component
  RealT* vx2;  ///< Nonmortar nodal velocities, x-component
  RealT* vy2;  ///< Nonmortar nodal velocities, y-component
  RealT* vz2;  ///< Nonmortar nodal velocities, z-component
  RealT* x;    ///< Nodal coordinates, x-component
  RealT* y;    ///< Nodal coordinates, y-component
  RealT* z;    ///< Nodal coordinates, z-component

  // CSR storage
  int* I;       ///< Offsets for CSR storage
  int* J;       ///< Column indices for all nonzero entries in CSR storage
  RealT* vals;  ///< Array of nonzero values

  RealT* gaps;       ///< Array of nodal gaps
  RealT* pressures;  ///< Array of nodal pressures

  RealT* mortar_bulk_mod;
  RealT* mortar_element_thickness;
  RealT* nonmortar_bulk_mod;
  RealT* nonmortar_element_thickness;

  bool registered_velocities1{ false };
  bool registered_velocities2{ false };
  bool mesh_constructed{ false };

 public:
  RealT* getX() const { return x; }
  RealT* getY() const { return y; }
  RealT* getZ() const { return z; }
  IndexT* getMortarFaceConnectivity() const { return faceConn1; }
  IndexT* getNonmortarFaceConnectivity() const { return faceConn2; }

  int getNumTotalNodes() const { return numTotalNodes; }

 public:
  /// Needed for initial shroud interface
  int getMortarFaceConnectivitySize() const { return numNodesPerFace * numMortarFaces; }
  int getNonmortarFaceConnectivitySize() const { return numNodesPerFace * numNonmortarFaces; }

 protected:
  // NONE
};

}  // namespace tribol

namespace mfem_ext {

/// Simple central difference method with homogeneous velocity BCs.
class CentralDiffSolver : public mfem::SecondOrderODESolver {
 public:
  /**
   * @brief Construct a new central difference solver object
   *
   * Supports homogeneous velocity BCs
   *
   * @param bc_vdofs_ List of vdofs to set homogeneous velocity BCs
   */
  CentralDiffSolver( const mfem::Array<int>& bc_vdofs_ );

  /**
   * @brief Updates x and dxdt after taking a step of size dt
   *
   * @param x Displacement vector
   * @param dxdt Velocity vector
   * @param t Current time
   * @param dt Timestep size
   */
  void Step( mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt ) override;

 private:
  /**
   * @brief Acceleration vector
   */
  mfem::Vector accel;

  /**
   * @brief List of vdofs to apply homogeneous velocity BCs
   */
  mfem::Array<int> bc_vdofs;

  /**
   * @brief Tracks whether a step has been taken yet
   */
  bool first_step;
};

#ifdef TRIBOL_USE_MPI
/// Explicit solid mechanics update with lumped mass
class ExplicitMechanics : public mfem::SecondOrderTimeDependentOperator {
 public:
  /**
   * @brief Construct a new explicit mechanics operator (elasticity, lumped
   * mass, and external force vector)
   *
   * @param fespace Finite element space of the primary fields
   * @param rho Density
   * @param lambda Lame constant
   * @param mu Lame constant
   */
  ExplicitMechanics( mfem::ParFiniteElementSpace& fespace, mfem::Coefficient& rho, mfem::Coefficient& lambda,
                     mfem::Coefficient& mu );

  using mfem::SecondOrderTimeDependentOperator::Mult;

  /**
   * @brief Compute acceleration given displacement and velocity
   *
   * @param u Displacement vector
   * @param dudt Velocity vector
   * @param a Acceleration vector
   */
  void Mult( const mfem::Vector& u, const mfem::Vector& dudt, mfem::Vector& a ) const override;

  /**
   * @brief Computes inverse lumped mass matrix
   *
   * @param fespace FE space of displacement field
   * @param rho Density coefficient
   * @return mfem::Vector holding mass
   */
  static mfem::Vector ComputeInvMass( mfem::ParFiniteElementSpace& fespace, mfem::Coefficient& rho );

  /**
   * @brief External force contribution (must be manually updated)
   */
  mfem::ParGridFunction f_ext;

 private:
  /**
   * @brief Elasticity bilinear form
   */
  mfem::ParBilinearForm elasticity;

  /**
   * @brief Inverse lumped mass matrix
   */
  mfem::Vector inv_lumped_mass;
};
#endif /* defined(TRIBOL_USE_MPI) */

}  // namespace mfem_ext

#endif /* SRC_TRIBOL_UTILS_TESTUTILS_HPP_ */
