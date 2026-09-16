#ifndef TRIBOL_TESTS_SPEC_SUPPORTEDMETHODS_HPP_
#define TRIBOL_TESTS_SPEC_SUPPORTEDMETHODS_HPP_

#include "tribol/method/Method.hpp"

namespace tribol::test::spec {

using PointwisePenalty = Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid,
                                constraint::Pointwise, enforcement::Penalty<stiffness::Constant, rate::None>,
                                response::Frictionless, formulation::PointwiseTraction, linearization::Exact>;

using PointwiseMaterialRateViscous =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<stiffness::Material, rate::Percentage>, response::ViscousTangential,
           formulation::PointwiseTraction, linearization::Exact>;

using PointwiseTiedNormal =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<>, response::TiedNormal, formulation::PointwiseTraction, linearization::Exact>;

using PointwiseTiedFull =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<>, response::TiedFull, formulation::PointwiseTraction, linearization::Exact>;

using ProjectedMultiplier = Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>,
                                   constraint::Nodal<basis::Dual>, enforcement::LagrangeMultiplier,
                                   response::Frictionless, formulation::WeightedWeakForm, linearization::Exact>;

using ConformingMultiplier = Method<geometry::ConformingOverlap, integration::Face<2>, constraint::Nodal<basis::Dual>,
                                    enforcement::LagrangeMultiplier, response::Frictionless,
                                    formulation::WeightedWeakForm, linearization::Exact>;

using DiagnosticWeights =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Dual>,
           enforcement::None, response::Frictionless, formulation::DiagnosticWeights, linearization::Exact>;

using VariationalNodalPenalty =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::Penalty<>, response::Frictionless, formulation::Variational, linearization::Exact>;

using VariationalQuadraturePenalty =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::Frictionless, formulation::Variational, linearization::Exact>;

using VariationalMultiplier =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::LagrangeMultiplier, response::Frictionless, formulation::Variational, linearization::Exact>;

using VariationalExternalPressure =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::Nodal<basis::Primal>,
           enforcement::ExternalPressure, response::Frictionless, formulation::Variational, linearization::Exact>;

}  // namespace tribol::test::spec

#endif
