#ifndef TRIBOL_TESTS_SPEC_SUPPORTEDMETHODS_HPP_
#define TRIBOL_TESTS_SPEC_SUPPORTEDMETHODS_HPP_

#include "tribol/method/Method.hpp"

namespace tribol::test::spec {

template <StiffnessPolicy Stiffness = stiffness::Constant, RatePolicy Rate = rate::None,
          ResponsePolicy Response = response::Frictionless>
using PointwiseFamily =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<Stiffness, Rate>, Response, formulation::PointwiseTraction, linearization::Exact>;

using PointwisePenalty = PointwiseFamily<>;

using PointwiseMaterialRateViscous =
    PointwiseFamily<stiffness::Material, rate::Percentage, response::ViscousTangential>;

using PointwiseTiedNormal =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<>, response::TiedNormal, formulation::PointwiseTraction, linearization::Exact>;

using PointwiseTiedFull =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<>, response::TiedFull, formulation::PointwiseTraction, linearization::Exact>;

using ProjectedMultiplier = Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>,
                                   constraint::Nodal<basis::Primal>, enforcement::LagrangeMultiplier,
                                   response::Frictionless, formulation::WeightedWeakForm, linearization::Exact>;

using ConformingMultiplier = Method<geometry::ConformingOverlap, integration::Face<2>, constraint::Nodal<basis::Primal>,
                                    enforcement::LagrangeMultiplier, response::Frictionless,
                                    formulation::WeightedWeakForm, linearization::Exact>;

using DiagnosticWeights =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::None, response::Frictionless, formulation::DiagnosticWeights, linearization::Exact>;

using VariationalNodalPenalty =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::Penalty<>, response::Frictionless, formulation::Variational, linearization::Exact>;

using VariationalQuadraturePenalty =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::Frictionless, formulation::Variational, linearization::Exact>;

using SmoothedVariationalNodalPenalty =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::SmoothedSegment<3>,
           constraint::Nodal<basis::Primal>, enforcement::Penalty<>, response::Frictionless, formulation::Variational,
           linearization::Exact>;

using SmoothedVariationalQuadraturePenalty =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::SmoothedSegment<3>,
           constraint::QuadraturePoint, enforcement::Penalty<>, response::Frictionless, formulation::Variational,
           linearization::Exact>;

using VariationalMultiplier =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::LagrangeMultiplier, response::Frictionless, formulation::Variational, linearization::Exact>;

using VariationalExternalPressure =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::ExternalPressure, response::Frictionless, formulation::Variational, linearization::Exact>;

}  // namespace tribol::test::spec

#endif
