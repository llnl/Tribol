// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/VirtualRefinementOperator.hpp"

#include "axom/slic.hpp"

namespace redecomp {

namespace {

mfem::IntegrationPoint MakePoint( double x, double y = 0.0 )
{
  mfem::IntegrationPoint point;
  point.Set2( x, y );
  return point;
}

}  // namespace

VirtualRefinementOperator::VirtualRefinementOperator( const mfem::FiniteElement& source_element, int vector_dimension,
                                                      int refinement_factor, mfem::Ordering::Type source_ordering )
    : mfem::Operator(),
      source_element_{ source_element },
      vector_dimension_{ vector_dimension },
      source_ordering_{ source_ordering },
      refinement_factor_{ refinement_factor }
{
  SLIC_ERROR_ROOT_IF( dynamic_cast<const mfem::NodalFiniteElement*>( &source_element_ ) == nullptr,
                      "VirtualRefinementOperator requires a nodal finite element." );
  SLIC_ERROR_ROOT_IF( vector_dimension_ <= 0, "VirtualRefinementOperator requires a positive vector dimension." );
  SLIC_ERROR_ROOT_IF( refinement_factor_ <= 0, "VirtualRefinementOperator requires a positive refinement factor." );
  BuildChildVertexReferencePoints();
  BuildEvaluationMatrix();
  width = source_element_.GetDof() * vector_dimension_;
  height = child_vertex_reference_points_.GetNPoints() * vector_dimension_;
}

void VirtualRefinementOperator::BuildChildVertexReferencePoints()
{
  const auto geometry = source_element_.GetGeomType();
  if ( geometry == mfem::Geometry::SEGMENT ) {
    num_children_ = refinement_factor_;
    vertices_per_child_ = 2;
    child_vertex_reference_points_.SetSize( num_children_ * vertices_per_child_ );
    int point = 0;
    for ( int child = 0; child < refinement_factor_; ++child ) {
      child_vertex_reference_points_.IntPoint( point++ ) =
          MakePoint( static_cast<double>( child ) / refinement_factor_ );
      child_vertex_reference_points_.IntPoint( point++ ) =
          MakePoint( static_cast<double>( child + 1 ) / refinement_factor_ );
    }
    return;
  }

  if ( geometry == mfem::Geometry::SQUARE ) {
    num_children_ = refinement_factor_ * refinement_factor_;
    vertices_per_child_ = 4;
    child_vertex_reference_points_.SetSize( num_children_ * vertices_per_child_ );
    int point = 0;
    for ( int j = 0; j < refinement_factor_; ++j ) {
      for ( int i = 0; i < refinement_factor_; ++i ) {
        const double x0 = static_cast<double>( i ) / refinement_factor_;
        const double x1 = static_cast<double>( i + 1 ) / refinement_factor_;
        const double y0 = static_cast<double>( j ) / refinement_factor_;
        const double y1 = static_cast<double>( j + 1 ) / refinement_factor_;
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x0, y0 );
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x1, y0 );
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x1, y1 );
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x0, y1 );
      }
    }
    return;
  }

  if ( geometry == mfem::Geometry::TRIANGLE ) {
    num_children_ = refinement_factor_ * refinement_factor_;
    vertices_per_child_ = 3;
    child_vertex_reference_points_.SetSize( num_children_ * vertices_per_child_ );
    int point = 0;
    for ( int j = 0; j < refinement_factor_; ++j ) {
      for ( int i = 0; i + j < refinement_factor_; ++i ) {
        const double x0 = static_cast<double>( i ) / refinement_factor_;
        const double x1 = static_cast<double>( i + 1 ) / refinement_factor_;
        const double y0 = static_cast<double>( j ) / refinement_factor_;
        const double y1 = static_cast<double>( j + 1 ) / refinement_factor_;
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x0, y0 );
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x1, y0 );
        child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x0, y1 );
        if ( i + j + 1 < refinement_factor_ ) {
          child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x1, y0 );
          child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x1, y1 );
          child_vertex_reference_points_.IntPoint( point++ ) = MakePoint( x0, y1 );
        }
      }
    }
    SLIC_ERROR_ROOT_IF( point != child_vertex_reference_points_.GetNPoints(),
                        "Virtual triangle refinement generated an unexpected number of child vertices." );
    return;
  }

  SLIC_ERROR_ROOT( "VirtualRefinementOperator supports segments, triangles, and quadrilaterals." );
}

void VirtualRefinementOperator::BuildEvaluationMatrix()
{
  const int num_source_dofs = source_element_.GetDof();
  const int num_child_vertices = child_vertex_reference_points_.GetNPoints();
  evaluation_matrix_.SetSize( num_child_vertices, num_source_dofs );
  mfem::Vector shape( num_source_dofs );
  for ( int vertex = 0; vertex < num_child_vertices; ++vertex ) {
    source_element_.CalcShape( child_vertex_reference_points_.IntPoint( vertex ), shape );
    for ( int source_dof = 0; source_dof < num_source_dofs; ++source_dof ) {
      evaluation_matrix_( vertex, source_dof ) = shape[source_dof];
    }
  }
}

void VirtualRefinementOperator::Mult( const mfem::Vector& source_element_values,
                                      mfem::Vector& child_vertex_values ) const
{
  SLIC_ERROR_ROOT_IF( source_element_values.Size() != Width(), "Virtual refinement source size mismatch." );
  SLIC_ERROR_ROOT_IF( child_vertex_values.Size() != Height(), "Virtual refinement destination size mismatch." );
  const int num_source_dofs = source_element_.GetDof();
  const int num_child_vertices = child_vertex_reference_points_.GetNPoints();
  for ( int component = 0; component < vector_dimension_; ++component ) {
    for ( int vertex = 0; vertex < num_child_vertices; ++vertex ) {
      double value = 0.0;
      for ( int source_dof = 0; source_dof < num_source_dofs; ++source_dof ) {
        const int source_index = source_ordering_ == mfem::Ordering::byNODES
                                     ? source_dof + num_source_dofs * component
                                     : component + vector_dimension_ * source_dof;
        value += evaluation_matrix_( vertex, source_dof ) * source_element_values[source_index];
      }
      child_vertex_values[vertex + num_child_vertices * component] = value;
    }
  }
}

void VirtualRefinementOperator::MultTranspose( const mfem::Vector& child_vertex_dual,
                                               mfem::Vector& source_element_dual ) const
{
  source_element_dual = 0.0;
  AddMultTranspose( child_vertex_dual, source_element_dual );
}

void VirtualRefinementOperator::AddMultTranspose( const mfem::Vector& child_vertex_dual,
                                                  mfem::Vector& source_element_dual, const mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( child_vertex_dual.Size() != Height(), "Virtual refinement child dual size mismatch." );
  SLIC_ERROR_ROOT_IF( source_element_dual.Size() != Width(), "Virtual refinement source dual size mismatch." );
  const int num_source_dofs = source_element_.GetDof();
  const int num_child_vertices = child_vertex_reference_points_.GetNPoints();
  for ( int component = 0; component < vector_dimension_; ++component ) {
    for ( int source_dof = 0; source_dof < num_source_dofs; ++source_dof ) {
      double value = 0.0;
      for ( int vertex = 0; vertex < num_child_vertices; ++vertex ) {
        value += evaluation_matrix_( vertex, source_dof ) * child_vertex_dual[vertex + num_child_vertices * component];
      }
      const int source_index = source_ordering_ == mfem::Ordering::byNODES ? source_dof + num_source_dofs * component
                                                                           : component + vector_dimension_ * source_dof;
      source_element_dual[source_index] += scale * value;
    }
  }
}

}  // namespace redecomp
