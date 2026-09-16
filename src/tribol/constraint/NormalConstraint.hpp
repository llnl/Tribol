#ifndef TRIBOL_CONSTRAINT_NORMALCONSTRAINT_HPP_
#define TRIBOL_CONSTRAINT_NORMALCONSTRAINT_HPP_

#include "tribol/basis/LinearBasis.hpp"
#include "tribol/integration/Quadrature.hpp"
#include "tribol/search/Search.hpp"

#include <array>
#include <type_traits>

namespace tribol::constraint {

template <typename Scalar>
struct NormalConstraintSampleT {
  std::array<Scalar, 3> mortar_position{};
  std::array<Scalar, 3> nonmortar_position{};
  std::array<Scalar, 3> normal{};
  basis::ShapeValuesT<Scalar> mortar_trial{};
  basis::ShapeValuesT<Scalar> nonmortar_trial{};
  basis::ShapeValuesT<Scalar> mortar_test{};
  Scalar gap{};
  Scalar weight{};
};

using NormalConstraintSample = NormalConstraintSampleT<Real>;

namespace detail {

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline Scalar normalGap( const std::array<Scalar, 3>& mortar,
                                                          const std::array<Scalar, 3>& nonmortar,
                                                          const std::array<Scalar, 3>& normal, int dimension )
{
  Scalar gap{};
  for ( int component = 0; component < dimension; ++component ) {
    gap += ( mortar[component] - nonmortar[component] ) * normal[component];
  }
  return gap;
}

}  // namespace detail

template <ConstraintPolicy Constraint, typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline NormalConstraintSampleT<Scalar> stageNormalConstraint(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair, const std::array<Scalar, 3>& mortar_position,
    const std::array<Scalar, 3>& nonmortar_position, const std::array<Scalar, 3>& normal, Scalar weight )
{
  NormalConstraintSampleT<Scalar> sample;
  sample.mortar_position = mortar_position;
  sample.nonmortar_position = nonmortar_position;
  sample.normal = normal;
  sample.weight = weight;
  sample.gap = detail::normalGap( mortar_position, nonmortar_position, normal, surfaces.mortar.dimension );
  const auto mortar_reference = basis::mapToReference( surfaces.mortar, pair.mortar_element, mortar_position );
  const auto nonmortar_reference =
      basis::mapToReference( surfaces.nonmortar, pair.nonmortar_element, nonmortar_position );
  sample.mortar_trial = basis::primalShape( surfaces.mortar.topologies[pair.mortar_element], mortar_reference );
  sample.nonmortar_trial =
      basis::primalShape( surfaces.nonmortar.topologies[pair.nonmortar_element], nonmortar_reference );
  if constexpr ( std::same_as<Constraint, Pointwise> || std::same_as<Constraint, QuadraturePoint> ) {
    sample.mortar_test = sample.mortar_trial;
  } else {
    using Basis = typename Constraint::basis_policy;
    sample.mortar_test = basis::shape<Basis>( surfaces.mortar.topologies[pair.mortar_element], mortar_reference );
  }
  return sample;
}

template <typename Scalar>
TRIBOL_HOST_DEVICE inline void addWeightedGap( const NormalConstraintSampleT<Scalar>& sample,
                                               ArrayView<Scalar> weighted_gap )
{
  const int count = sample.mortar_test.size;
  for ( int node = 0; node < count; ++node ) {
    weighted_gap[node] += sample.weight * sample.mortar_test[node] * sample.gap;
  }
}

template <typename Scalar>
TRIBOL_HOST_DEVICE inline void addMortarWeights( const NormalConstraintSampleT<Scalar>& sample,
                                                 ArrayView<Scalar> weights )
{
  const int count = sample.mortar_test.size;
  for ( int node = 0; node < count; ++node ) {
    for ( int trial = 0; trial < sample.nonmortar_trial.size; ++trial ) {
      weights[node * sample.nonmortar_trial.size + trial] +=
          sample.weight * sample.mortar_test[node] * sample.nonmortar_trial[trial];
    }
  }
}

static_assert( std::is_trivially_copyable_v<NormalConstraintSample> );

}  // namespace tribol::constraint

#endif
