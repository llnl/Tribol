// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_TRIBOLASSEMBLEFORCE_HPP_
#define SRC_TRIBOL_PHYSICS_TRIBOLASSEMBLEFORCE_HPP_

#include "tribol/physics/ForceAndGapMethod.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

#include "axom/slic.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace tribol {

/**
 * @brief Concrete implementation of ForceAndGapMethod that assembles contributions on-rank
 *
 * This class orchestrates the calculation of element-level contributions via the
 * ElementForceAndGap policy and assembles them into global arrays and an MFEM SparseMatrix.
 *
 * @tparam ElementForceAndGap Policy class defining element-level physics (gap, force, stiffness)
 * @tparam IntegrationRule Integration rule policy
 * @tparam PointwiseGapAndNormal Gap and normal computation policy
 */
template <typename ElementForceAndGap, typename IntegrationRule, typename PointwiseGapAndNormal>
class TribolAssembleForce : public ForceAndGapMethod<IntegrationRule, PointwiseGapAndNormal> {
 public:
  /**
   * @brief Main Constructor with mandatory mesh and force data, and optional derivatives/gaps/pressures
   */
  TribolAssembleForce( MeshData& mesh1, MeshData& mesh2, ArrayViewT<RealT> forces1, ArrayViewT<RealT> forces2,
                       ElementForceAndGap&& element_force_and_gap, std::shared_ptr<ArrayViewT<RealT>> pressures,
                       std::shared_ptr<ArrayViewT<RealT>> gaps, std::shared_ptr<mfem::SparseMatrix> df1_dx1,
                       std::shared_ptr<mfem::SparseMatrix> df1_dx2, std::shared_ptr<mfem::SparseMatrix> df2_dx1,
                       std::shared_ptr<mfem::SparseMatrix> df2_dx2, std::shared_ptr<mfem::SparseMatrix> dg_dx1,
                       std::shared_ptr<mfem::SparseMatrix> dg_dx2, std::shared_ptr<mfem::SparseMatrix> df1_dp,
                       std::shared_ptr<mfem::SparseMatrix> df2_dp )
      : mesh1_( mesh1 ),
        mesh2_( mesh2 ),
        pressures_( pressures ),
        gaps_( gaps ),
        forces1_( forces1 ),
        forces2_( forces2 ),
        df1_dx1_( df1_dx1 ),
        df1_dx2_( df1_dx2 ),
        df2_dx1_( df2_dx1 ),
        df2_dx2_( df2_dx2 ),
        dg_dx1_( dg_dx1 ),
        dg_dx2_( dg_dx2 ),
        df1_dp_( df1_dp ),
        df2_dp_( df2_dp ),
        element_force_and_gap_( std::move( element_force_and_gap ) ),
        energy_( 0.0 )
  {
  }

  /**
   * @brief Constructor for Force contributions only
   */
  TribolAssembleForce( MeshData& mesh1, MeshData& mesh2, ArrayViewT<RealT> forces1, ArrayViewT<RealT> forces2,
                       ElementForceAndGap&& element_force_and_gap )
      : TribolAssembleForce( mesh1, mesh2, forces1, forces2, std::move( element_force_and_gap ), nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr )
  {
  }

  /**
   * @brief Constructor for Force and Stiffness contributions
   */
  TribolAssembleForce( MeshData& mesh1, MeshData& mesh2, ArrayViewT<RealT> forces1, ArrayViewT<RealT> forces2,
                       ElementForceAndGap&& element_force_and_gap, std::shared_ptr<mfem::SparseMatrix> df1_dx1,
                       std::shared_ptr<mfem::SparseMatrix> df1_dx2, std::shared_ptr<mfem::SparseMatrix> df2_dx1,
                       std::shared_ptr<mfem::SparseMatrix> df2_dx2,
                       std::shared_ptr<ArrayViewT<RealT>> pressures = nullptr,
                       std::shared_ptr<mfem::SparseMatrix> df1_dp = nullptr,
                       std::shared_ptr<mfem::SparseMatrix> df2_dp = nullptr )
      : TribolAssembleForce( mesh1, mesh2, forces1, forces2, std::move( element_force_and_gap ), pressures, nullptr,
                             df1_dx1, df1_dx2, df2_dx1, df2_dx2, nullptr, nullptr, df1_dp, df2_dp )
  {
  }

  /**
   * @brief Constructor for Force and Gap contributions
   */
  TribolAssembleForce( MeshData& mesh1, MeshData& mesh2, ArrayViewT<RealT> forces1, ArrayViewT<RealT> forces2,
                       ElementForceAndGap&& element_force_and_gap, std::shared_ptr<ArrayViewT<RealT>> gaps,
                       std::shared_ptr<mfem::SparseMatrix> dg_dx1 = nullptr,
                       std::shared_ptr<mfem::SparseMatrix> dg_dx2 = nullptr )
      : TribolAssembleForce( mesh1, mesh2, forces1, forces2, std::move( element_force_and_gap ), nullptr, gaps, nullptr,
                             nullptr, nullptr, nullptr, dg_dx1, dg_dx2, nullptr, nullptr )
  {
  }

  void updateNodalGaps( IntegrationRule& integration_rule, PointwiseGapAndNormal& gap_method ) override
  {
    if ( !gaps_ && !dg_dx1_ && !dg_dx2_ ) {
      return;
    }

    MeshData::Viewer mesh1_view = mesh1_.getView();
    MeshData::Viewer mesh2_view = mesh2_.getView();

    int dim = mesh1_view.spatialDimension();
    int n1 = mesh1_view.numberOfNodesPerElement();
    int n2 = mesh2_view.numberOfNodesPerElement();

    std::vector<RealT> coords1( n1 * dim );
    std::vector<RealT> coords2( n2 * dim );

    const auto& pairs = integration_rule.getPairs();
    for ( const auto& pair : pairs ) {
      // Get coordinates
      mesh1_view.getFaceCoords( pair.m_element_id1, coords1.data() );
      mesh2_view.getFaceCoords( pair.m_element_id2, coords2.data() );

      mfem::Vector elem_gaps;
      mfem::DenseMatrix elem_dg_dx1;
      mfem::DenseMatrix elem_dg_dx2;

      if ( dg_dx1_ || dg_dx2_ ) {
        element_force_and_gap_.computeGap( pair, integration_rule, gap_method, coords1, coords2, elem_gaps, elem_dg_dx1,
                                           elem_dg_dx2 );
      } else {
        element_force_and_gap_.computeGap( pair, integration_rule, gap_method, coords1, coords2, elem_gaps );
      }

      if ( gaps_ ) {
        assembleVector( pair, mesh1_view, elem_gaps, *gaps_ );
      }

      if ( dg_dx1_ ) {
        assembleMatrix11( pair, mesh1_view, elem_dg_dx1, *dg_dx1_ );
      }
      if ( dg_dx2_ ) {
        assembleMatrixMixed( pair, mesh1_view, mesh2_view, elem_dg_dx2, *dg_dx2_ );
      }
    }
  }

  void updateNodalForces( IntegrationRule& integration_rule, PointwiseGapAndNormal& gap_method ) override
  {
    MeshData::Viewer mesh1_view = mesh1_.getView();
    MeshData::Viewer mesh2_view = mesh2_.getView();

    int dim = mesh1_view.spatialDimension();
    int n1 = mesh1_view.numberOfNodesPerElement();
    int n2 = mesh2_view.numberOfNodesPerElement();

    std::vector<RealT> coords1( n1 * dim );
    std::vector<RealT> coords2( n2 * dim );

    // Reset energy
    energy_ = 0.0;

    const auto& pairs = integration_rule.getPairs();
    for ( const auto& pair : pairs ) {
      mesh1_view.getFaceCoords( pair.m_element_id1, coords1.data() );
      mesh2_view.getFaceCoords( pair.m_element_id2, coords2.data() );

      mfem::Vector elem_f1, elem_f2;
      mfem::DenseMatrix elem_df1_dx1, elem_df1_dx2, elem_df2_dx1, elem_df2_dx2;
      mfem::DenseMatrix elem_df1_dp, elem_df2_dp;

      // Assuming ElementForceAndGap also takes pressures if needed?
      // Usually computeForce depends on current pressure (Lambda).
      // If pressure is state-dependent, we might need to pass element pressures.
      // But prompt just said "take element coordinate vectors".
      // I'll stick to coords.

      RealT elem_energy = 0.0;
      if ( df1_dx1_ || df1_dx2_ || df2_dx1_ || df2_dx2_ || df1_dp_ || df2_dp_ ) {
        element_force_and_gap_.computeForce( pair, integration_rule, gap_method, coords1, coords2, pressures_, elem_f1,
                                             elem_f2, elem_df1_dx1, elem_df1_dx2, elem_df2_dx1, elem_df2_dx2,
                                             elem_df1_dp, elem_df2_dp, elem_energy );
      } else {
        element_force_and_gap_.computeForce( pair, integration_rule, gap_method, coords1, coords2, pressures_, elem_f1,
                                             elem_f2, elem_energy );
      }

      energy_ += elem_energy;

      // Assemble Forces
      assembleVector( pair, mesh1_view, elem_f1, forces1_ );
      assembleVector( pair, mesh2_view, elem_f2, forces2_ );

      // Assemble Stiffness
      if ( df1_dx1_ ) {
        assembleMatrix11( pair, mesh1_view, elem_df1_dx1, *df1_dx1_ );
      }
      if ( df1_dx2_ ) {
        assembleMatrixMixed( pair, mesh1_view, mesh2_view, elem_df1_dx2, *df1_dx2_ );
      }
      if ( df2_dx1_ ) {
        assembleMatrixMixed( pair, mesh2_view, mesh1_view, elem_df2_dx1, *df2_dx1_ );
      }
      if ( df2_dx2_ ) {
        assembleMatrix22( pair, mesh2_view, elem_df2_dx2, *df2_dx2_ );
      }

      // Assemble Pressure Derivatives
      if ( df1_dp_ ) {
        assembleMatrixMixed( pair, mesh1_view, mesh2_view, elem_df1_dp, *df1_dp_ );
      }
      if ( df2_dp_ ) {
        assembleMatrix22( pair, mesh2_view, elem_df2_dp, *df2_dp_ );
      }
    }
  }

  RealT computeTimeStep( IntegrationRule& integration_rule, PointwiseGapAndNormal& gap_method ) override
  {
    return std::numeric_limits<RealT>::max();
  }

#ifdef BUILD_REDECOMP
  void getMfemForce( mfem::Vector& forces ) const override
  {
    SLIC_ERROR_ROOT( "getMfemForce should not be called with TribolAssembleForce." );
  }

  void getMfemGap( mfem::Vector& gaps ) const override
  {
    SLIC_ERROR_ROOT( "getMfemGap should not be called with TribolAssembleForce." );
  }

  mfem::ParGridFunction& getMfemPressure() override
  {
    SLIC_ERROR_ROOT( "getMfemPressure should not be called with TribolAssembleForce." );
    static mfem::ParGridFunction dummy;
    return dummy;
  }

  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override
  {
    SLIC_ERROR_ROOT( "getMfemDfDx should not be called with TribolAssembleForce." );
    return nullptr;
  }
  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override
  {
    SLIC_ERROR_ROOT( "getMfemDgDx should not be called with TribolAssembleForce." );
    return nullptr;
  }
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override
  {
    SLIC_ERROR_ROOT( "getMfemDfDp should not be called with TribolAssembleForce." );
    return nullptr;
  }
#endif

  RealT getEnergy() const { return energy_; }

 private:
  void getSubVector( const mfem::Vector& src, int offset, int size, mfem::Vector& dest ) const
  {
    dest.SetSize( size );
    for ( int i = 0; i < size; ++i ) {
      dest( i ) = src( offset + i );
    }
  }

  void getSubMatrix( const mfem::DenseMatrix& src, int r_off, int r_size, int c_off, int c_size,
                     mfem::DenseMatrix& dest ) const
  {
    dest.SetSize( r_size, c_size );
    for ( int i = 0; i < r_size; ++i ) {
      for ( int j = 0; j < c_size; ++j ) {
        dest( i, j ) = src( r_off + i, c_off + j );
      }
    }
  }

  void getGlobalDofs( int element_id, const MeshData::Viewer& mesh, mfem::Array<int>& dofs, int dim_stride ) const
  {
    int n = mesh.numberOfNodesPerElement();
    dofs.SetSize( n * dim_stride );
    for ( int i = 0; i < n; ++i ) {
      int node = mesh.getGlobalNodeId( element_id, i );
      for ( int d = 0; d < dim_stride; ++d ) {
        dofs[i * dim_stride + d] = node * dim_stride + d;
      }
    }
  }

  void assembleVector( const InterfacePair& pair, const MeshData::Viewer& mesh, const mfem::Vector& elem_vec,
                       ArrayViewT<RealT>& global_vec )
  {
    int element_id = ( mesh.meshId() == mesh1_.getView().meshId() ) ? pair.m_element_id1 : pair.m_element_id2;
    int stride = ( elem_vec.Size() == mesh.numberOfNodesPerElement() ) ? 1 : mesh.spatialDimension();

    mfem::Array<int> dofs;
    getGlobalDofs( element_id, mesh, dofs, stride );
    for ( int i = 0; i < dofs.Size(); ++i ) {
      if ( dofs[i] >= 0 && dofs[i] < global_vec.size() ) {
        global_vec[dofs[i]] += elem_vec( i );
      }
    }
  }

  void assembleMatrix( int row_elem_id, int col_elem_id, const MeshData::Viewer& row_mesh,
                       const MeshData::Viewer& col_mesh, const mfem::DenseMatrix& elem_mat,
                       mfem::SparseMatrix& global_mat )
  {
    int row_stride = elem_mat.Height() / row_mesh.numberOfNodesPerElement();
    int col_stride = elem_mat.Width() / col_mesh.numberOfNodesPerElement();

    mfem::Array<int> row_dofs, col_dofs;
    getGlobalDofs( row_elem_id, row_mesh, row_dofs, row_stride );
    getGlobalDofs( col_elem_id, col_mesh, col_dofs, col_stride );

    global_mat.AddSubMatrix( row_dofs, col_dofs, const_cast<mfem::DenseMatrix&>( elem_mat ) );
  }

  // Specific wrappers to match main loop
  void assembleMatrix11( const InterfacePair& pair, const MeshData::Viewer& mesh1, const mfem::DenseMatrix& mat,
                         mfem::SparseMatrix& global )
  {
    assembleMatrix( pair.m_element_id1, pair.m_element_id1, mesh1, mesh1, mat, global );
  }
  void assembleMatrix22( const InterfacePair& pair, const MeshData::Viewer& mesh2, const mfem::DenseMatrix& mat,
                         mfem::SparseMatrix& global )
  {
    assembleMatrix( pair.m_element_id2, pair.m_element_id2, mesh2, mesh2, mat, global );
  }
  void assembleMatrixMixed( const InterfacePair& pair, const MeshData::Viewer& meshRow, const MeshData::Viewer& meshCol,
                            const mfem::DenseMatrix& mat, mfem::SparseMatrix& global )
  {
    int row_id = ( meshRow.meshId() == mesh1_.getView().meshId() ) ? pair.m_element_id1 : pair.m_element_id2;
    int col_id = ( meshCol.meshId() == mesh1_.getView().meshId() ) ? pair.m_element_id1 : pair.m_element_id2;
    assembleMatrix( row_id, col_id, meshRow, meshCol, mat, global );
  }

  MeshData& mesh1_;
  MeshData& mesh2_;

  std::shared_ptr<ArrayViewT<RealT>> pressures_;
  std::shared_ptr<ArrayViewT<RealT>> gaps_;
  ArrayViewT<RealT> forces1_;
  ArrayViewT<RealT> forces2_;

  std::shared_ptr<mfem::SparseMatrix> df1_dx1_;
  std::shared_ptr<mfem::SparseMatrix> df1_dx2_;
  std::shared_ptr<mfem::SparseMatrix> df2_dx1_;
  std::shared_ptr<mfem::SparseMatrix> df2_dx2_;
  std::shared_ptr<mfem::SparseMatrix> dg_dx1_;
  std::shared_ptr<mfem::SparseMatrix> dg_dx2_;
  std::shared_ptr<mfem::SparseMatrix> df1_dp_;
  std::shared_ptr<mfem::SparseMatrix> df2_dp_;

  ElementForceAndGap element_force_and_gap_;

  RealT energy_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_TRIBOLASSEMBLEFORCE_HPP_ */
