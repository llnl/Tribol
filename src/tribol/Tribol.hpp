#ifndef TRIBOL_TRIBOL_HPP_
#define TRIBOL_TRIBOL_HPP_

#include "tribol/adapters/array/ArrayContact.hpp"
#include "tribol/basis/LinearBasis.hpp"
#include "tribol/constraint/NormalConstraint.hpp"
#include "tribol/contact/Contact.hpp"
#include "tribol/core/ArrayView.hpp"
#include "tribol/core/Config.hpp"
#include "tribol/core/MeshView.hpp"
#include "tribol/core/Version.hpp"
#include "tribol/evaluation/State.hpp"
#include "tribol/execution/ContactCuda.hpp"
#include "tribol/evaluation/PolicyEvaluator.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/execution/CudaPenalty.hpp"
#include "tribol/execution/PenaltyKernel.hpp"
#include "tribol/geom/ConformingOverlap.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/integration/InteractionQuadrature.hpp"
#include "tribol/integration/Quadrature.hpp"
#include "tribol/method/Method.hpp"
#include "tribol/method/Traits.hpp"
#include "tribol/method/Validation.hpp"
#include "tribol/search/Search.hpp"

#endif
