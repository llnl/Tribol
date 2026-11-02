// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Traverse geometry-approved contact pairs and gather current coordinates without generating quadrature.
 */
#ifndef SRC_TRIBOL_INTEG_CONTACTDOMAIN_HPP_
#define SRC_TRIBOL_INTEG_CONTACTDOMAIN_HPP_

#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include <stdexcept>

namespace tribol {

/**
 * @brief Borrow geometry-approved pairs and current host mesh views for one contact calculation.
 *
 * Pairs are visited in their supplied order, including pairs that contribute no overlap or active contact. Coordinates
 * are gathered on each visit; no geometry, quadrature, or calculated fields are cached. Pair IDs must identify valid
 * elements in the supplied meshes. Mesh arrays and pairs must remain alive, host-accessible, and unchanged throughout
 * traversal. Destroy this view before mesh re-registration or decomposition updates.
 * @tparam FirstGeometry First-element geometry defining Coords and spatial_dimension.
 * @tparam SecondGeometry Second-element geometry defining Coords and spatial_dimension.
 */
template <typename FirstGeometry, typename SecondGeometry>
class ContactDomain {
 public:
  static constexpr IndexT first_dofs = FirstGeometry::Coords::size();    ///< First-element coordinate count.
  static constexpr IndexT second_dofs = SecondGeometry::Coords::size();  ///< Second-element coordinate count.
  static constexpr IndexT dof_count = first_dofs + second_dofs;          ///< Packed pair coordinate count.
  using Coordinates = MdArray<RealT, dof_count>;                         ///< Pair coordinates owned during each visit.
  using CoordinateView = MdSpan<const RealT, dof_count>;                 ///< Borrowed pair coordinates.

  /**
   * @brief Borrow current geometry and a supplied sequence of accepted pairs; allocate no storage.
   * @param mesh1 Current first mesh; mortar methods use it as the non-mortar side.
   * @param mesh2 Current second mesh; mortar methods use it as the mortar side.
   * @param pairs Geometry-approved element pairs; no additional search or filtering is performed.
   * @throws std::invalid_argument Mesh dimensions or node counts do not match the geometry coordinate layouts.
   */
  ContactDomain( MeshData::Viewer mesh1, MeshData::Viewer mesh2, ArrayViewT<const InterfacePair> pairs )
      : mesh1_( mesh1 ), mesh2_( mesh2 ), pairs_( pairs )
  {
    validateMesh<FirstGeometry>( mesh1_ );
    validateMesh<SecondGeometry>( mesh2_ );
  }

  /** @brief Read the first mesh view. @return Borrowed view with the same lifetime as the domain inputs. */
  MeshData::Viewer mesh1() const { return mesh1_; }
  /** @brief Read the second mesh view. @return Borrowed view with the same lifetime as the domain inputs. */
  MeshData::Viewer mesh2() const { return mesh2_; }
  /** @brief Count supplied pairs, including inactive ones. @return Number of pairs visited by forEachPair(). */
  IndexT size() const { return pairs_.size(); }

  /**
   * @brief Gather each pair's coordinates on the stack and call the visitor in supplied pair order.
   *
   * Coordinates are packed by element, node, then spatial component. The coordinate view is valid only during the
   * visitor call and must not be retained. An empty domain does not call the visitor. Exceptions propagate to the
   * caller.
   * @tparam Visitor Callable taking (InterfacePair, CoordinateView).
   * @param visitor Operation applied to each pair; may evaluate and differentiate a definition using the coordinates.
   */
  template <typename Visitor>
  void forEachPair( Visitor&& visitor ) const
  {
    for ( const auto& pair : pairs_ ) {
      Coordinates coordinates;
      mesh1_.getFaceCoords( pair.element_id1_, coordinates.data() );
      mesh2_.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
      visitor( pair, CoordinateView{ coordinates.data(), dof_count } );
    }
  }

 private:
  /**
   * @brief Check the coordinate layout before gathering into fixed-size stack storage.
   * @tparam Geometry Geometry providing Coords and spatial_dimension.
   * @param mesh Current mesh whose coordinate layout is checked.
   * @throws std::invalid_argument The mesh has an incompatible dimension or element node count.
   */
  template <typename Geometry>
  static void validateMesh( MeshData::Viewer mesh )
  {
    if ( mesh.spatialDimension() != Geometry::spatial_dimension ||
         mesh.numberOfNodesPerElement() != Geometry::Coords::template extent<0>() ) {
      throw std::invalid_argument( "Contact domain mesh layouts must match the pair geometries." );
    }
  }

  MeshData::Viewer mesh1_;                 ///< Borrowed first mesh arrays for the current calculation.
  MeshData::Viewer mesh2_;                 ///< Borrowed second mesh arrays for the current calculation.
  ArrayViewT<const InterfacePair> pairs_;  ///< Borrowed geometry-approved pairs, in traversal order.
};

}  // namespace tribol
#endif  // SRC_TRIBOL_INTEG_CONTACTDOMAIN_HPP_
