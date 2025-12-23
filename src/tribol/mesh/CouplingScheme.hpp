// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
#ifndef SRC_TRIBOL_MESH_COUPLINGSCHEME_HPP_
#define SRC_TRIBOL_MESH_COUPLINGSCHEME_HPP_

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/utils/DataManager.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/geom/CompGeom.hpp"
#include "tribol/geom/ElementNormal.hpp"

#ifdef TRIBOL_USE_ENZYME
#include "tribol/geom/NodalNormal.hpp"
#endif

// Axom includes
#include "axom/core.hpp"

namespace tribol {
// Struct to hold on-rank coupling scheme face-pair reporting data
// generated from computational geometry issues
struct PairReportingData {
 public:
  int numBadOrientation{ 0 };
  int numBadOverlaps{ 0 };
  int numBadFaceGeometry{ 0 };
};

/**
 * @brief Enumerates execution mode errors
 */
enum class ExecutionModeError
{
  UNKNOWN_MEMORY_SPACE,
  NON_MATCHING_MEMORY_SPACE,
  BAD_MODE_FOR_MEMORY_SPACE,
  INCOMPATIBLE_METHOD,
  NO_ERROR
};

// Helper struct to handle coupling scheme errors
struct CouplingSchemeErrors {
 public:
  ModeError cs_mode_error;
  CaseError cs_case_error;
  MethodError cs_method_error;
  ModelError cs_model_error;
  EnforcementError cs_enforcement_error;
  EnforcementDataErrors cs_enforcement_data_error;
  ExecutionModeError cs_execution_mode_error;

  void printModeErrors();
  void printCaseErrors();
  void printMethodErrors();
  void printModelErrors();
  void printEnforcementErrors();
  void printEnforcementDataErrors();
  void printExecutionModeErrors();
};

/**
 * @brief Enumerates execution mode informational messages
 */
enum class ExecutionModeInfo
{
  NONOPTIMAL_MODE_FOR_MEMORY_SPACE,
  NO_INFO
};

// Helper struct to handle coupling scheme infomational messages
struct CouplingSchemeInfo {
 public:
  // Add info enums as needed
  CaseInfo cs_case_info;
  EnforcementInfo cs_enforcement_info;
  ExecutionModeInfo cs_execution_mode_info;

  void printCaseInfo();
  void printEnforcementInfo();
  void printExecutionModeInfo();
};

/*!
 * \brief The CouplingScheme class defines the coupling between two meshes
 *  in the computational domain.
 *
 * A CouplingScheme defines the physics mode, method, enforcement, model
 * and binning for the coupling. It also holds the list of interacting mesh
 * entities (e.g. surface/element combinations) that result from the binning
 * and geometric checks.
 *
 *  \see InterfacePairs
 *  \see ContactMode
 *  \see ContactMethod
 *  \see ContactModel
 *  \see EnforcementMethod
 *  \see BinningMethod
 *  \see CouplingSchemeManager
 */
class CouplingScheme {
 public:
  /**
   * @brief Nested class for holding views (non-owned, shallow copies) of coupling scheme data
   */
  class Viewer {
   public:
    /**
     * @brief Construct a new CouplingScheme::Viewer object
     *
     * @param cs CouplingScheme to create a view of
     */
    Viewer( CouplingScheme& cs );

    /**
     * @brief Spatial dimension of the meshes in the coupling scheme
     *
     * @return spatial dimension
     */
    TRIBOL_HOST_DEVICE int spatialDimension() const { return m_mesh1.spatialDimension(); }

    /**
     * @brief Return a view of the first mesh in the coupling scheme
     *
     * @return view of first mesh
     */
    TRIBOL_HOST_DEVICE const MeshData::Viewer& getMesh1View() const { return m_mesh1; }

    /**
     * @brief Return a view of the second mesh in the coupling scheme
     *
     * @return view of second mesh
     */
    TRIBOL_HOST_DEVICE const MeshData::Viewer& getMesh2View() const { return m_mesh2; }

    /**
     * @brief Get the struct defining enforcement options
     *
     * @return const reference to EnforcementOptions struct
     */
    TRIBOL_HOST_DEVICE const EnforcementOptions& getEnforcementOptions() const { return m_enforcement_options; }

    /**
     * @brief Get the contact mode
     *
     * @return const reference to the contact mode
     */
    TRIBOL_HOST_DEVICE ContactMode getContactMode() const { return m_contact_mode; }

    /**
     * @brief Get the parameters struct
     *
     * @return a const reference to the parameters
     */
    TRIBOL_HOST_DEVICE const Parameters& getParameters() const { return m_parameters; }

    /**
     * @brief Get the effective binning proximity scale
     *
     * @return the effective binning proximity scale used on the coupling scheme
     */
    TRIBOL_HOST_DEVICE RealT getEffectiveBinningProximityScale() const { return m_effective_binning_proximity_scale; }

    /**
     * @brief Return the contact plane given by id
     *
     * @param id identifier for a contact plane
     * @return contact plane object
     */
    TRIBOL_HOST_DEVICE ContactPlanePair& getContactPlanePair( IndexT id ) const;

    /**
     * @brief Get the timestep scale
     *
     * @return timestep scale
     */
    TRIBOL_HOST_DEVICE RealT getTimestepScale() const { return m_parameters.timestep_scale; }

    /**
     * @brief Get the gap tolerance that determines in contact face-pairs
     *
     * @return the gap tolerance for the common plane method
     */
    TRIBOL_HOST_DEVICE RealT getGapTol( int fid1, int fid2 ) const;

    /**
     * @brief Get a view of the computational geometry container
     *
     * @return a comp geom view
     */
    TRIBOL_HOST_DEVICE const CompGeom::Viewer& getCompGeomView() const { return m_cg_pairs; }

    /**
     * @brief Perform interface pair pruning based on the contact method
     *
     * \param [in] fid1 id of the first face
     * \param [in] fid2 id of the second face
     *
     * \return true if interface pair is to be pruned (excluded); false if the pair is a contact candidate
     */
    TRIBOL_HOST_DEVICE bool pruneMethodFacePair( const IndexT fid1, const IndexT fid2 ) const;

   private:
    /// Struct holding parameters for the coupling scheme
    Parameters m_parameters;

    RealT m_effective_binning_proximity_scale;

    /// Defines the contact mode
    ContactMode m_contact_mode;

    /// Defines the contact case: special algorithmic considerations
    ContactCase m_contact_case;

    /// Defines the contact method
    ContactMethod m_contact_method;

    /// Struct holding the enforcement options for the contact method
    EnforcementOptions m_enforcement_options;

    /// View of the first mesh
    MeshData::Viewer m_mesh1;

    /// View of the second mesh
    MeshData::Viewer m_mesh2;

    /// View of computational geometry container
    CompGeom::Viewer m_cg_pairs;

  };  // end class CouplingScheme::Viewer

  /**
   * @brief Default constructor. Disabled.
   */
  CouplingScheme() = delete;

  /**
   * @brief Creates a CouplingScheme instance between a pair of meshes
   *
   * @param [in] cs_id coupling scheme id
   * @param [in] mesh_id1 id of the first contact surface mesh (corresponds to mortar surface with a mortar method)
   * @param [in] mesh_id2 id of the second contact surface mesh (corresponds to nonmortar surface with a mortar method),
   * or ANY_MESH for multiple meshes
   * @param [in] contact_mode the type of contact, e.g. SURFACE_TO_SURFACE
   * @param [in] contact_case the specific case of contact application, e.g. auto
   * @param [in] contact_method the contact method, e.g. SINGLE_MORTAR
   * @param [in] contact_model the contact model, e.g. FRICTION_COULOMB
   * @param [in] enforcement_method the enforcement method, e.g. PENALTY
   * @param [in] binning_method the binning method, e.g. BINNING_GRID
   *
   * Per-cycle rebinning is enabled by default.
   */
  CouplingScheme( IndexT cs_id, IndexT mesh_id1, IndexT mesh_id2, int contact_mode, int contact_case,
                  int contact_method, int contact_model, int enforcement_method, int binning_method,
                  ExecutionMode given_exec_mode = ExecutionMode::Dynamic );

  // Prevent copying
  CouplingScheme( const CouplingScheme& other ) = delete;
  CouplingScheme& operator=( const CouplingScheme& other ) = delete;
  // Enable moving
  CouplingScheme( CouplingScheme&& other ) = default;
  CouplingScheme& operator=( CouplingScheme&& other ) = default;

  /**
   * @brief Returns the problem communicator
   *
   * @return problem mpi communicator
   */
  CommT getProblemComm() { return m_problem_comm; }

  /**
   * @brief Get the ID of the coupling scheme
   *
   * @return unique coupling scheme ID
   */
  int getId() const { return m_id; }

  /**
   * @brief Get the integer ID of the first mesh
   *
   * @return unique ID of the first mesh
   */
  int getMeshId1() const { return m_mesh_id1; }

  /**
   * @brief Get the integer ID of the second mesh
   *
   * @return unique ID of the second mesh
   */
  int getMeshId2() const { return m_mesh_id2; }

  /**
   * @brief Get the Parameters struct
   *
   * @return reference to the Parameters struct
   */
  Parameters& getParameters() { return m_parameters; }

  /**
   * @brief Get a reference to the first mesh
   *
   * @return MeshData reference
   */
  MeshData& getMesh1() { return *m_mesh1; }

  /// @overload
  const MeshData& getMesh1() const { return *m_mesh1; }

  /**
   * @brief Get a reference to the second mesh
   *
   * @return MeshData reference
   */
  MeshData& getMesh2() { return *m_mesh2; }

  /// @overload
  const MeshData& getMesh2() const { return *m_mesh2; }

  /**
   * @brief Get a reference to the computational geometry container
   *
   * @return CompGeom reference
   */
  const CompGeom& getCompGeom() const { return m_cg_pairs; }

  /**
   * @brief Get the execution mode for the coupling scheme
   *
   * @return ExecutionMode
   */
  ExecutionMode getExecutionMode() const { return m_exec_mode; }

  /**
   * @brief Get the Umpire allocator ID for mesh data (zero if built without Umpire)
   *
   * @return allocator ID
   */
  int getAllocatorId() const { return m_allocator_id; }

  int getNumTotalNodes() const { return m_numTotalNodes; }

  /**
   * @brief Get the contact mode (pairing of mesh types)
   *
   * @return ContactMode
   */
  ContactMode getContactMode() const { return m_contactMode; }

  /**
   * @brief Get the contact case (special algorithmic considerations for method)
   *
   * @return ContactCase
   */
  ContactCase getContactCase() const { return m_contactCase; }

  /**
   * @brief Get the contact method (algorithm to integrate contact weak form term)
   *
   * @return ContactMethod
   */
  ContactMethod getContactMethod() const { return m_contactMethod; }

  /**
   * @brief Get the contact model (constitutive modeling options)
   *
   * @return ContactModel
   */
  ContactModel getContactModel() const { return m_contactModel; }

  /**
   * @brief Get the enforcement method (defines enforcement scheme for contact constraints)
   *
   * @return EnforcementMethod
   */
  EnforcementMethod getEnforcementMethod() const { return m_enforcementMethod; }

  /**
   * @brief Get the spatial binning method
   *
   * @return BinningMethod
   */
  BinningMethod getBinningMethod() const { return m_binningMethod; }

  /**
   * @brief Set the spatial binning method
   *
   * @param binningMethod new enum value
   */
  void setBinningMethod( BinningMethod binningMethod ) { m_binningMethod = binningMethod; }

  /**
   * @brief Get the method data for the contact method
   *
   * @return MethodData pointer
   */
  MethodData* getMethodData() const { return m_methodData.get(); }

  /**
   * @brief Get the enforcement options for the enforcement method
   *
   * @return reference to the EnforcementOptions struct
   */
  EnforcementOptions& getEnforcementOptions() { return m_enforcementOptions; }

  /// @overload
  const EnforcementOptions& getEnforcementOptions() const { return m_enforcementOptions; }

  /**
   * @brief Get struct holding errors during coupling scheme initialization
   *
   * @return CouplingSchemeErrors reference
   */
  CouplingSchemeErrors& getCouplingSchemeErrors() { return m_couplingSchemeErrors; }

  /**
   * @brief Get struct holding informational messages that happened during coupling scheme initialization
   *
   * @return CouplingSchemeInfo& reference
   */
  CouplingSchemeInfo& getCouplingSchemeInfo() { return m_couplingSchemeInfo; }

  /**
   * @brief Construct a non-owned, shallow copy of the CouplingScheme
   *
   * @return CouplingScheme::Viewer type
   */
  CouplingScheme::Viewer getView() { return *this; }

  /**
   * @brief Spatial dimension of the mesh
   *
   * @return spatial dimension
   */
  int spatialDimension() const
  {
    // same for both meshes since meshes are required to have the same element
    // types
    return m_mesh1->spatialDimension();
  }

  /**
   * @brief Disable/enable per-cycle rebinning of interface pairs
   *
   * @param [in] pred True to disable rebinning, false otherwise
   */
  void setFixedBinning( bool pred ) { m_fixedBinning = pred; }

  /**
   * @brief Disable/Enable per-cycle rebinning of interface pairs based on
   * contact mode
   */
  void setFixedBinningPerCase()
  {
    if ( m_isBinned && m_contactCase == NO_SLIDING ) {
      m_fixedBinning = true;
    }
  }

  /**
   * @brief Set the MPI communicator for the coupling scheme
   *
   * @param comm MPI communicator
   */
  void setMPIComm( CommT comm ) { m_problem_comm = comm; }

  /**
   * @brief Check whether the coupling scheme has been binned
   *
   * @return true if the binning has occurred, otherwise false
   */
  bool isBinned() const { return m_isBinned; }

  /**
   * @brief Check whether the coupling scheme is using tied contact
   */
  bool isTied() const { return m_isTied; }

  /**
   * @brief Check if per-cycle rebinning is disabled
   *
   * @return true if the binning is fixed, false, if the binning method requires
   * per-cycle rebinning
   */
  bool hasFixedBinning() const { return m_fixedBinning; }

  /**
   * @brief Returns a reference to the associated InterfacePairs
   *
   * @return reference to the InterfacePairs array
   */
  ArrayT<InterfacePair>& getInterfacePairs() { return m_interface_pairs; }

  /// @overload
  const ArrayT<InterfacePair>& getInterfacePairs() const { return m_interface_pairs; }

  /**
   * @brief Get the number of active pairs on the coupling scheme
   *
   * @note an active interface pair is a proximate contact candidate with an
   * associated contact plane. this includes face pairs in contact and in
   * separation up to some criteria.
   *
   * @return number of active interface pairs
   */
  int getNumActivePairs() const { return m_cg_pairs.getNumActivePairs( getContactMethod() ); }

  /**
   * @brief Return the contact plane given by id
   *
   * @param id identifier for a contact plane
   * @return contact plane object
   */
  const ContactPlanePair& getContactPlanePair( IndexT id ) const;

  /**
   * @brief Set whether the coupling scheme has been binned
   *
   * @param [in] pred True to indicate binning has occurred
   */
  void setBinned( bool pred ) { m_isBinned = pred; }

  /**
   * @brief Set the stored mesh pointers based on the two mesh IDs provided
   *
   * @return True if meshes are registered; false otherwise
   */
  bool setMeshPointers();

  /**
   * @brief Returns true if a valid coupling scheme, otherwise false
   *
   * @return bool indicating if coupling scheme is valid
   */
  bool isValidCouplingScheme();

  /**
   * @brief Returns true if one or both meshes are zero-element, null meshes
   *
   * @return true if one or both null meshes in coupling scheme
   */
  bool nullMeshes() const { return m_nullMeshes; }

  /**
   * @brief Returns true if a valid mode is specified, otherwise false
   *
   * @return true indicating if the mode is valid
   */
  bool isValidMode();

  /**
   * @brief Returns true if a valid case is specified, otherwise false
   *
   * @return true indicating if the case is valid
   */
  bool isValidCase();

  /**
   * @brief Returns true if a valid method is specified, otherwise false
   *
   * @return true indicating if the method is valid
   */
  bool isValidMethod();

  /**
   * @brief Returns true if a valid model is specified, otherwise false
   *
   * @return true indicating if the model is valid
   */
  bool isValidModel();

  /**
   * @brief Returns true if a valid enforcement is specified, otherwise false
   *
   * @return true indicating if the enforcement is valid
   */
  bool isValidEnforcement();

  /**
   * @brief Check for correct enforcement data for a given method
   *
   * @return 0 for correct enforcement data, 1 otherwise
   */
  int checkEnforcementData();

  /**
   * @brief Check for correct execution mode for given memory spaces
   *
   * @return 0 for valid execution mode, 1 for invalid execution mode, 2 for
   * execution mode related informational messages
   */
  int checkExecutionModeData();

  /**
   * @brief Initializes the coupling scheme
   *
   * @return true if the coupling scheme was successfully initialized
   */
  bool init();

  /**
   * @brief Allocate method data on the coupling scheme
   */
  void allocateMethodData();

  /**
   * @brief Performs the binning between mesh 1 and mesh 2
   */
  void performBinning();

  /**
   * @brief Applies the CouplingScheme
   *
   * @param [in] cycle the cycle at which this method is invoked.
   * @param [in] t the simulation time at the given cycle
   * @param [in/out] dt the simulation dt at the given cycle sent back as Tribol timestep vote
   *
   * @return 0 if successful apply
   */
  int apply( int cycle, RealT t, RealT& dt );

  /**
   * @brief Wrapper around method specific calculation of the Tribol timestep vote
   *
   * @param [in/out] dt simulation timestep at given cycle
   */
  void computeTimeStep( RealT& dt );

  /**
   * @brief Set the output directory for file output
   *
   * @param directory string giving a file system path
   */
  void setOutputDirectory( const std::string& directory ) { m_output_directory = directory; }

  /**
   * @brief Wrapper to call method specific visualization output routines
   *
   * @param [in] dir the registered output directory
   * @param [in] v_type visualization option type
   * @param [in] cycle simulation cycle
   * @param [in] t simulation time at given cycle
   */
  void writeInterfaceOutput( const std::string& dir, const VisType v_type, const int cycle, const RealT t );

  /**
   * @brief Sets the coupling scheme logging level member variable
   *
   * @param [in] log_level the LoggingLevel enum value
   */
  void setLoggingLevel( const LoggingLevel log_level ) { m_loggingLevel = log_level; }

  /**
   * @brief Sets the SLIC logging level per the coupling scheme logging level
   *
   * @pre must call setLoggingLevel() first
   */
  void setSlicLoggingLevel();

  /**
   * @brief Get the SLIC logging level active for the coupling scheme
   *
   * @return LoggingLevel
   */
  LoggingLevel getLoggingLevel() const { return m_loggingLevel; }

  /**
   * @brief This updates the total number of types of face geometry exceptions
   *
   * @pre The face_exception is generated by calling CheckInterfacePair()
   */
  void updatePairReportingData( const FaceGeomException face_exception );

  /**
   * @brief This debug prints the total number of types of face geometry errors
   */
  void printPairReportingData();

  /**
   * @brief Get the effective binning proximity scale
   *
   * @note The effective binning proximity scale is the user supplied binning proximity scaled by the LOR factor. Thus,
   * the effective binning proximity is independent of the low-order mesh when a higher-order mesh is used with the MFEM
   * interface. This is generally the binning proximity that should be used in binning and geometric filtering.
   *
   * @pre CouplingScheme::init() must be called prior
   *
   * @return effective binning proximity scale
   */
  RealT getEffectiveBinningProximityScale() const { return m_effective_binning_proximity_scale; }

  /**
   * @brief Enables Enzyme AD for exact Jacobian calculations
   *
   * @param useEnzyme Turns on Enzyme support if true
   */
  void enableEnzyme( bool useEnzyme );

  /**
   * @brief Is Enzyme AD enabled for exact Jacobian calculations?
   *
   * @return true Yes
   * @return false No
   */
  bool isEnzymeEnabled() const { return m_useEnzyme; }

  /**
   * @brief Create MethodData to save Jacobian contributions related to a nodally defined normal
   */
  void createNodalNormalJacobianData();

  /**
   * @brief Get the method data for the derivative of the force w.r.t. the normal
   *
   * @return MethodData pointer
   */
  MethodData* getDfDnMethodData() const { return m_dfdnJacobian.get(); }

  /**
   * @brief Get the method data for the derivative of the normal w.r.t. the nodal coordinates
   *
   * @return MethodData pointer
   */
  MethodData* getDnDxMethodData() const { return m_dndxJacobian.get(); }

#ifdef BUILD_REDECOMP

  /**
   * @brief Check if coupling scheme has MFEM mesh data
   *
   * MFEM mesh data includes the MFEM volume mesh, transfer operators to move
   * mesh data from the MFEM volume mesh to the Tribol surface mesh, and mesh
   * data such as displacement, velocity, and force (response).
   *
   * @return true: MFEM mesh data exists
   * @return false: MFEM mesh data does not exist
   */
  bool hasMfemData() const { return m_mfemMeshData != nullptr; }

  /**
   * @brief Get the MFEM mesh data object
   *
   * MFEM mesh data includes the MFEM volume mesh, transfer operators to move
   * mesh data from the MFEM volume mesh to the Tribol surface mesh, and mesh
   * data such as displacement, velocity, and force (response).
   *
   * @return MfemMeshData*
   */
  MfemMeshData* getMfemMeshData() { return m_mfemMeshData.get(); }

  /**
   * @brief Get the MFEM mesh data object (const overload)
   *
   * MFEM mesh data includes the MFEM volume mesh, transfer operators to move mesh data
   * from the MFEM volume mesh to the Tribol surface mesh, and mesh data such as
   * displacement, velocity, and force (response).
   *
   * @return const MfemMeshData*
   */
  const MfemMeshData* getMfemMeshData() const { return m_mfemMeshData.get(); }

  /**
   * @brief Sets the MFEM mesh data object
   *
   * MFEM mesh data includes the MFEM volume mesh, transfer operators to move
   * mesh data from the MFEM volume mesh to the Tribol surface mesh, and mesh
   * data such as displacement, velocity, and force (response).
   *
   * @param mfemMeshData Unique pointer to MFEM mesh data
   */
  void setMfemMeshData( std::unique_ptr<MfemMeshData> mfemMeshData ) { m_mfemMeshData = std::move( mfemMeshData ); }

  /**
   * @brief Check if coupling scheme has MFEM submesh field data
   *
   * MFEM submesh field data includes a parent-linked boundary mfem::ParSubMesh,
   * transfer operators to move mesh data from the boundary submesh to the
   * Tribol surface mesh, and submesh data such as gap and pressure.
   *
   * @return true: MFEM submesh field data exists
   * @return false: MFEM submesh field data does not exist
   */
  bool hasMfemSubmeshData() const { return m_mfemSubmeshData != nullptr; }

  /**
   * @brief Get the MFEM submesh field data object
   *
   * MFEM submesh field data includes a parent-linked boundary mfem::ParSubMesh,
   * transfer operators to move mesh data from the boundary submesh to the
   * Tribol surface mesh, and submesh data such as gap and pressure.
   *
   * @return MfemSubmeshData*
   */
  MfemSubmeshData* getMfemSubmeshData() { return m_mfemSubmeshData.get(); }

  /**
   * @brief Get the MFEM submesh field data object (const overload)
   *
   * MFEM submesh field data includes a parent-linked boundary mfem::ParSubMesh,
   * transfer operators to move mesh data from the boundary submesh to the
   * Tribol surface mesh, and submesh data such as gap and pressure.
   *
   * @return const MfemSubmeshData*
   */
  const MfemSubmeshData* getMfemSubmeshData() const { return m_mfemSubmeshData.get(); }

  /**
   * @brief Sets the MFEM submesh field data object
   *
   * MFEM submesh field data includes a parent-linked boundary mfem::ParSubMesh,
   * transfer operators to move mesh data from the boundary submesh to the
   * Tribol surface mesh, and submesh data such as gap and pressure.
   *
   * @param MfemSubmeshData Unique pointer to MFEM submesh field data
   */
  void setMfemSubmeshData( std::unique_ptr<MfemSubmeshData> mfemSubmeshData )
  {
    m_mfemSubmeshData = std::move( mfemSubmeshData );
  }

  /**
   * @brief Check if coupling scheme has MFEM Jacobian data
   *
   * MFEM Jacobian data includes transfer operators to move Jacobian
   * contributions from the Tribol surface mesh to the MFEM parent mesh and
   * parent-linked boundary submesh.
   *
   * @return true: MFEM Jacobian data exists
   * @return false: MFEM Jacobian data does not exist
   */
  bool hasMfemJacobianData() const { return m_mfemJacobianData != nullptr; }

  /**
   * @brief Get the MFEM Jacobian data object
   *
   * MFEM Jacobian data includes transfer operators to move Jacobian
   * contributions from the Tribol surface mesh to the MFEM parent mesh and
   * parent-linked boundary submesh.
   *
   * @return MfemJacobianData*
   */
  MfemJacobianData* getMfemJacobianData() { return m_mfemJacobianData.get(); }

  /**
   * @brief Get the MFEM jacobian data object (const overload)
   *
   * MFEM jacobian data includes transfer operators to move Jacobian
   * contributions from the Tribol surface mesh to the MFEM parent mesh and
   * parent-linked boundary submesh.
   *
   * @return MfemJacobianData*
   */
  const MfemJacobianData* getMfemJacobianData() const { return m_mfemJacobianData.get(); }

  /**
   * @brief Sets the MFEM jacobian data object
   *
   * MFEM jacobian data includes transfer operators to move Jacobian
   * contributions from the Tribol surface mesh to the MFEM parent mesh and
   * parent-linked boundary submesh.
   *
   * @param mfemJacobianData Unique pointer to MFEM jacobian data
   */
  void setMfemJacobianData( std::unique_ptr<MfemJacobianData> mfemJacobianData )
  {
    m_mfemJacobianData = std::move( mfemJacobianData );
  }

#endif /* BUILD_REDECOMP */

  /**
   * @brief Computes common-plane specific time step vote
   *
   * @param [in/out] dt simulation timestep at given cycle
   */
  void computeCommonPlaneTimeStep( RealT& dt );

 private:
  CommT m_problem_comm = TRIBOL_COMM_WORLD;  ///! MPI communicator for the problem

  IndexT m_id;  ///< Coupling Scheme id

  IndexT m_mesh_id1;  ///< Integer id for mesh 1
  IndexT m_mesh_id2;  ///< Integer id for mesh 2

  MeshData* m_mesh1;  ///< Pointer to mesh 1 (reset every time init() is called)
  MeshData* m_mesh2;  ///< Pointer to mesh 2 (reset every time init() is called)

  ExecutionMode m_given_exec_mode;  ///< User preferred execution mode (set by constructor)

  ExecutionMode m_exec_mode;  ///< Execution mode for kernels (set when init() is called)
  int m_allocator_id;         ///< Allocator for arrays used in kernels (set when init() is called)

  Parameters m_parameters;              ///< Struct holding coupling scheme parameters
  std::string m_output_directory = "";  ///< Output directory for visualization dumps

  bool m_nullMeshes{ false };  ///< True if one or both meshes are zero-element (null) meshes
  bool m_isValid{ true };      ///< False if the coupling scheme is not valid per call to init()

  int m_numTotalNodes;  ///< Total number of nodes in the coupling scheme

  ContactMode m_contactMode;              ///< Contact mode
  ContactCase m_contactCase;              ///< Contact case
  ContactMethod m_contactMethod;          ///< Contact method
  ContactModel m_contactModel;            ///< Contact model
  EnforcementMethod m_enforcementMethod;  ///< Contact enforcement method
  BinningMethod m_binningMethod;          ///< Contact binning method

  LoggingLevel m_loggingLevel;  ///< logging level enum for coupling scheme

  bool m_fixedBinning;  ///< True if using fixed binning for all cycles
  bool m_isBinned;      ///< True if binning has occured
  bool m_isTied;        ///< True if surfaces have been "tied" (Tied contact only)

  bool m_useEnzyme = false;                    ///< Use Enzyme for Jacobian calculations
  std::unique_ptr<MethodData> m_dfdnJacobian;  ///< Store derivative of force w.r.t. normal on element pairs
  std::unique_ptr<MethodData> m_dndxJacobian;  ///< Store derivative of normal w.r.t. nodal coordinates on element pairs

  ArrayT<InterfacePair> m_interface_pairs;  ///< List of interface pairs

  CompGeom m_cg_pairs;  ///< Computational geometry container object

  std::unique_ptr<MethodData> m_methodData;  ///< method object holding required interface method data

  EnforcementOptions m_enforcementOptions;      ///< struct with options underneath chosen enforcement
  CouplingSchemeErrors m_couplingSchemeErrors;  ///< struct handling coupling scheme errors
  CouplingSchemeInfo m_couplingSchemeInfo;      ///< struct handling info to be printed

  PairReportingData m_pairReportingData;  ///< struct handling on-rank pair reporting data from computational geometry

  RealT m_effective_binning_proximity_scale;  ///< Binning proximity scaled by the LOR factor. Scaling by the LOR factor
                                              ///< makes proximity detection independent of the LOR factor and based on
                                              ///< the HO mesh when using the MFEM interface.

#ifdef BUILD_REDECOMP

  std::unique_ptr<MfemMeshData> m_mfemMeshData;          ///< MFEM mesh data
  std::unique_ptr<MfemSubmeshData> m_mfemSubmeshData;    ///< MFEM submesh field data
  std::unique_ptr<MfemJacobianData> m_mfemJacobianData;  ///< MFEM jacobian data

#endif /* BUILD_REDECOMP */

};  // end class CouplingScheme

using CouplingSchemeManager = DataManager<CouplingScheme>;

} /* namespace tribol */

#endif /* SRC_TRIBOL_MESH_COUPLINGSCHEME_HPP_ */
