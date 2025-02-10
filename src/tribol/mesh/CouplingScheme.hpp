// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
#ifndef TRIBOL_COUPLINGSCHEME_HPP_
#define TRIBOL_COUPLINGSCHEME_HPP_

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MfemData.hpp"

// Axom includes
#include "axom/core.hpp"

namespace tribol {

// Helper struct to handle coupling scheme errors
struct CouplingSchemeErrors {
 public:
  CouplingSchemeErrors(){};

  ~CouplingSchemeErrors(){};

  ModeError cs_mode_error;
  CaseError cs_case_error;
  MethodError cs_method_error;
  ModelError cs_model_error;
  EnforcementError cs_enforcement_error;
  EnforcementDataErrors cs_enforcement_data_error;

  void printModeErrors();
  void printCaseErrors();
  void printMethodErrors();
  void printModelErrors();
  void printEnforcementErrors();
  void printEnforcementDataErrors();
};

struct CouplingSchemeInfo {
 public:
  CouplingSchemeInfo(){};

  ~CouplingSchemeInfo(){};

  void printCaseInfo();
  void printEnforcementInfo();

  // Add info enums as needed
  CaseInfo cs_case_info;
  EnforcementInfo cs_enforcement_info;
};

// forward declaration
class MethodData;
class InterfacePairs;

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
  /*!
   * \brief Default constructor. Disabled.
   */
  CouplingScheme() = delete;

  /*!
   * \brief Creates a CouplingSchmeme instance between a pair of meshes
   *
   * \param [in] params pointer to the parameters object
   * \param [in] meshId1 ID of the mortar surface
   * \param [in] meshId2 ID of the nonmortar surface, or ANY_MESH for multiple meshes
   * \param [in] contact_mode the type of contact, e.g. SURFACE_TO_SURFACE
   * \param [in] contact_case the specific case of contact application, e.g. auto
   * \param [in] contact_method the contact method, e.g. SINGLE_MORTAR
   * \param [in] contact_model the contact model, e.g. COULOMB,
   * \param [in] enforcement_method the enforcement method, e.g. PENALTY
   * \param [in] binning_method the binning method, e.g. BINNING_GRID
   *
   * Per-cycle rebinning is enabled by default.
   */
  CouplingScheme( integer couplingSchemeId, integer meshId1, integer meshId2, integer contact_mode,
                  integer contact_case, integer contact_method, integer contact_model, integer enforcement_method,
                  integer binning_method );

  /*!
   * \brief Destructor.
   */
  ~CouplingScheme();

  /// \name Getters
  /// @{

  integer getMeshId1() const { return m_meshId1; }
  integer getMeshId2() const { return m_meshId2; }

  integer getId() const { return m_id; }

  integer getNumTotalNodes() const { return m_numTotalNodes; }

  ContactMode getContactMode() const { return m_contactMode; }
  ContactCase getContactCase() const { return m_contactCase; }
  ContactMethod getContactMethod() const { return m_contactMethod; }
  ContactModel getContactModel() const { return m_contactModel; }
  EnforcementMethod getEnforcementMethod() const { return m_enforcementMethod; }
  BinningMethod getBinningMethod() const { return m_binningMethod; }

  MethodData* getMethodData() const { return m_methodData; }

  // TODO test these getters
  EnforcementOptions& getEnforcementOptions() { return m_enforcementOptions; }
  const EnforcementOptions& getEnforcementOptions() const { return m_enforcementOptions; }

  CouplingSchemeErrors& getCouplingSchemeErrors() { return m_couplingSchemeErrors; }
  CouplingSchemeInfo& getCouplingSchemeInfo() { return m_couplingSchemeInfo; }

  integer spatialDimension() const
  {
    parameters_t& params = parameters_t::getInstance();
    return params.dimension;
  }

  /*!
   * Disable/Enable per-cycle rebinning of interface pairs
   *
   * \param [in] pred True to disable rebinning, false otherwise
   */
  void setFixedBinning( bool pred ) { m_fixedBinning = pred; }

  /*!
   * Disable/Enable per-cycle rebinning of interface pairs based
   * on contact mode
   *
   */
  void setFixedBinningPerCase()
  {
    if ( m_isBinned && m_contactCase == NO_SLIDING ) {
      m_fixedBinning = true;
    }
  }

  /*!
   * Check whether the coupling scheme has been binned
   *
   * \return True if the binning has occurred, otherwise false
   *
   */
  bool isBinned() const { return m_isBinned; }

  /*!
   * Check whether the coupling scheme is using tied contact
   */
  bool isTied() const { return m_isTied; }

  /*!
   * Check if per-cycle rebinning is disabled
   *
   * \return True if the binning is fixed, false, if the binning method
   * requires per-cycle rebinning.
   */
  bool hasFixedBinning() const { return m_fixedBinning; }

  /*!
   * \brief Returns a pointer to the associated InterfacePairs
   *
   * \return ptr pointer to the InterfacePairs instance.
   */
  InterfacePairs* getInterfacePairs() { return m_interfacePairs; }

  /*!
   * \brief Returns a pointer to the associated InterfacePairs
   *
   * \return ptr pointer to the InterfacePairs instance.
   */
  InterfacePairs* getInterfacePairs() const { return m_interfacePairs; }

  /// @}

  /*!
   * Get the number of active pairs on the coupling scheme
   *
   * \return number of active interface pairs
   */
  integer getNumActivePairs() const { return m_numActivePairs; }

  /*!
   * Get the gap tolerance that determines in contact face-pairs
   * for each supported interface method
   *
   * \return the gap tolerance based on the method
   */
  real getGapTol( int fid1, int fid2 ) const;

  /*!
   * Set whether the coupling scheme has been binned
   *
   * \param [in] pred True to indicate binning has occurred
   *
   */
  void setBinned( bool pred ) { m_isBinned = pred; }

  /*!
   * \brief Returns true if a valid coupling scheme, otherwise false
   *
   * \return bool indicating if coupling scheme is valid;
   */
  bool isValidCouplingScheme();

  /*!
   * \brief Returns true if one or both meshes are zero-element, null meshes
   *
   * \return true if one or both null meshes in coupling scheme
   */
  bool nullMeshes() { return m_nullMeshes; }

  /*!
   * \brief Returns true if one or both meshes are zero-element, null meshes
   *
   * \return true if one or both null meshes in coupling scheme
   */
  bool nullMeshes() const { return m_nullMeshes; }

  /*!
   * \brief Returns true if a valid mode is specified, otherwise false
   *
   * \return true indicating if the mode is valid;
   */
  bool isValidMode();

  /*!
   * \brief Returns true if a valid case is specified, otherwise false
   *
   * \return true indicating if the case is valid;
   */
  bool isValidCase();

  /*!
   * \brief Returns true if a valid method is specified, otherwise false
   *
   * \return true indicating if the method is valid;
   */
  bool isValidMethod();

  /*!
   * \brief Returns true if a valid model is specified, otherwise false
   *
   * \return true indicating if the model is valid;
   */
  bool isValidModel();

  /*!
   * \brief Returns true if a valid enforcement is specified, otherwise false
   *
   * \return true indicating if the enforcement is valid;
   */
  bool isValidEnforcement();

  /*!
   * \brief Check for correct enforcement data for a given method
   *
   * \return 0 for correct enforcement data, 1 otherwise
   */
  int checkEnforcementData();

  /*!
   * \brief Initializes the coupling scheme
   *
   * \return true if the coupling scheme was successfully initialized
   */
  bool init();

  /*!
   * \brief Allocate method data on the coupling scheme
   *
   */
  void allocateMethodData();

  /*!
   * \brief Performs the binning between mesh 1 and mesh 2
   *
   */
  void performBinning();

  /*!
   * \brief Applies the CouplingScheme
   *
   * \param [in] cycle the cycle at which this method is invoked.
   * \param [in] t the simulation time at the given cycle
   * \param [in/out] dt the simulation dt at the given cycle sent back as Tribol timestep vote
   *
   * \return 0 if successful apply
   *
   */
  int apply( integer cycle, real t, real& dt );

  /*!
   * \brief Wrapper around method specific calculation of the Tribol timestep vote
   *
   * \param [in/out] dt simulation timestep at given cycle
   *
   */
  void computeTimeStep( real& dt );

  /*!
   * \brief Wrapper to call method specific visualization output routines
   *
   * \param [in] dir the registered output directory
   * \param [in] v_type visualization option type
   * \param [in] cycle simulation cycle
   * \param [in] t simulation time at given cycle
   *
   */
  void writeInterfaceOutput( const std::string& dir, const VisType v_type, const integer cycle, const real t );

  /*!
   * \brief Sets the coupling scheme logging level member variable
   *
   * \param [in] log_level the LoggingLevel enum value
   *
   */
  void setLoggingLevel( const LoggingLevel log_level ) { m_loggingLevel = log_level; }

  /*!
   * \brief Sets the SLIC logging level per the coupling scheme logging level
   *
   * \pre must call setLoggingLevel() first
   *
   */
  void setSlicLoggingLevel();

  LoggingLevel getLoggingLevel() const { return m_loggingLevel; }

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

 private:
  /*!
   * \brief Computes common-plane specific time step vote
   *
   * \param [in/out] dt simulation timestep at given cycle
   *
   */
  void computeCommonPlaneTimeStep( real& dt );

 private:
  integer m_id;  ///< Coupling Scheme id

  integer m_meshId1;  ///< Integer id for mesh 1
  integer m_meshId2;  ///< Integer id for mesh 2

  bool m_nullMeshes{ false };  ///< True if one or both meshes are zero-element (null) meshes
  bool m_isValid{ true };      ///< False if the coupling scheme is not valid per call to init()

  integer m_numTotalNodes;  ///< Total number of nodes in the coupling scheme

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

  InterfacePairs* m_interfacePairs;  ///< List of interface pairs

  integer m_numActivePairs;  ///< number of active interface pairs from InterfacePairs list

  MethodData* m_methodData;  ///< method object holding required interface method data

  EnforcementOptions m_enforcementOptions;      ///< struct with options underneath chosen enforcement
  CouplingSchemeErrors m_couplingSchemeErrors;  ///< struct handling coupling scheme errors
  CouplingSchemeInfo m_couplingSchemeInfo;      ///< struct handling info to be printed

#ifdef BUILD_REDECOMP

  std::unique_ptr<MfemMeshData> m_mfemMeshData;          ///< MFEM mesh data
  std::unique_ptr<MfemSubmeshData> m_mfemSubmeshData;    ///< MFEM submesh field data
  std::unique_ptr<MfemJacobianData> m_mfemJacobianData;  ///< MFEM jacobian data

#endif /* BUILD_REDECOMP */

  DISABLE_COPY_AND_ASSIGNMENT( CouplingScheme );
  DISABLE_MOVE_AND_ASSIGNMENT( CouplingScheme );
};

} /* namespace tribol */

#endif /* SRC_MESH_COUPLINGSCHEME_HPP_ */
