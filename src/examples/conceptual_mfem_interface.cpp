/**
 * @file conceptual_mfem_interface.cpp
 *
 * Conceptual design for host code interactions with Tribol's MFEM interface
 */

#include <optional>

#include "mfem.hpp"

/**
 * @brief Scenario 1: Tribol as a geometry engine + conforming integration rule generator
 *
 * @param mesh Mesh to solve the problem on
 * @param fields Field data including current coords, ref coords, velocity, inverse nodal mass, acceleration, etc.; many
 * of these values will be optional (e.g. wrapped in std::optional)
 * @param options Sets the enforcement location, enforcement type, energy-based functional, quadrature, search,
 * smoothing, etc. (anything with an enum)
 * @param settings Sets surface boundary attributes, penalty, smoothing radius, search radius, etc. (anything
 * numerical); many of these values will be optional (e.g. wrapped in std::optional)
 * @return Array holding points and weights (and their displacement derivatives?) for all element pairs in contact
 */
auto makeIntegrationRule( mfem::ParMesh mesh, tribol::MfemFields fields, tribol::Options options,
                          tribol::Settings settings )
{
  // The integration rule generator has no stored integration rule. It simply computes it and returns it to the host
  // code.
  auto rule_gen = tribol::createIntegrationRuleGenerator( mesh, fields, options, settings );

  // These would be run in a loop for a non-linear solve/multiple timestep problem

  // Running this does the following:
  // 1. Transfers the latest field values to the redecomp mesh from the parent mesh fields and optionally re-builds the
  // redecomp mesh. The update_redecomp variable (passed in settings) will be std::optional and if it is not defined,
  // the redecomp mesh will be re-build based on the amount of mesh movement since the last rebuild. Likely called
  // updateFields() in the underlying class (TBD: should this be public?).
  // 2. Call the coarse and refined search routines for the rule generator, identifying a set of contact pairs that
  // likely project onto each other (but may not). Likely called updateSearch() in the underlying class (TBD: should
  // this be public?).
  // 3. Builds a new integration rule (set of integration points and weights on the parent element) for each identified
  // contact pair (+ derivatives w.r.t. displacement?). The host code could then use this integration rule to compute
  // their own integrals over a contact surface independent of Tribol. (TBD: should we remove pairs that have no
  // integration points?)
  return rule_gen.createIntegrationRule();
}

/**
 * @brief Scenario 2: Tribol as a geometry engine + conforming integration rule generator + active set identifier
 *
 * Useful for generating an integration rule + developing an active set (gap <= 0) on which to apply physics
 *
 * @param mesh Mesh to solve the problem on
 * @param fields Field data including current coords, ref coords, velocity, inverse nodal mass, acceleration, etc.; many
 * of these values will be optional (e.g. wrapped in std::optional)
 * @param options Sets the enforcement location, enforcement type, energy-based functional, quadrature, search,
 * smoothing, etc. (anything with an enum)
 * @param settings Sets surface boundary attributes, penalty, smoothing radius, search radius, etc. (anything
 * numerical); many of these values will be optional (e.g. wrapped in std::optional)
 */
auto makeIntegrationRuleWithGaps( mfem::ParMesh mesh, tribol::MfemFields fields, tribol::Options options,
                                  tribol::Settings settings )
{
  // The gap evaluator will derive from the integration rule class + add a gap enforcement mechanism. There are no
  // stored integration rules or gap values in the gap evaluator. It simply computes these values and returns them to
  // the host code.
  auto gap_eval = tribol::createGapEvaluator( mesh, fields, options, settings );

  // These would be run in a loop for a non-linear solve/multiple timestep problem

  // This does the same as Scenario 1 above.
  auto integ_rule = gap_eval.createIntegrationRule();

  // This computes the gaps at each node, provided the gap evaluation technique of the derived class defines a gap at
  // the nodes. The integ_rule argument will be std::optional; when run without it, the method will call
  // gap_eval.createIntegrationRule() internally.
  auto nodal_gaps = gap_eval.computeNodalGaps( integ_rule );

  // Optionally implemented quadrature point version of above. Only one of them would be implemented, depending on the
  // gap evaluator.
  auto quad_pt_gaps = gap_eval.computeQuadraturePointGaps( integ_rule );

  // In a host code, one would usually compute something with the computed rule and gaps inside a solver loop, but we'll
  // just return them as a pair in this conceptual doc.
  return std::make_pair( integ_rule, nodal_gaps );
}

/**
 * @brief Scenario 3: Tribol as a contact gap and force calculator (+ displacement derivatives)
 *
 * Useful for computing gaps, contact forces, and derivatives w.r.t displacement
 *
 * @param mesh Mesh to solve the problem on
 * @param fields Field data including current coords, ref coords, velocity, inverse nodal mass, acceleration, etc.; many
 * of these values will be optional (e.g. wrapped in std::optional)
 * @param options Sets the enforcement location, enforcement type, energy-based functional, quadrature, search,
 * smoothing, etc. (anything with an enum)
 * @param settings Sets surface boundary attributes, penalty, smoothing radius, search radius, etc. (anything
 * numerical); many of these values will be optional (e.g. wrapped in std::optional)
 */
auto computeContactData( mfem::ParMesh mesh, tribol::MfemFields fields, tribol::Options options,
                         tribol::Settings settings )
{
  // Contact formulation implementations will derive from the gap evaluator class + add a way to calculate forces (+
  // optionally energy and stiffness). There are no stored forces (or their derivatives) in the formulation. It simply
  // computes these values and returns them to the host code.
  auto formulation = tribol::createContactFormulation( mesh, fields, options, settings );

  // These would be run in a loop for a non-linear solve/multiple timestep problem

  // This does the same as Scenario 1 above. Compute integration rule once and re-use it in later evaluations for a
  // timestep/iteration.
  auto integ_rule = formulation.createIntegrationRule();

  // This does the same as Scenario 2 above. Stopping at the gaps allows pressures to be updated in a penalty
  // formulation enforced outside Tribol. Also returns derivatives w.r.t. displacement.
  auto nodal_gaps = formulation.computeNodalGaps( integ_rule );

  // Computes the forces at each node for the method defined in the formulation. Both integ_rule and nodal_gaps are
  // std::optional; if integ_rule is not provided, then formulation.createIntegrationRule() is run; if nodal_gaps is not
  // provided, then formulation.computeNodalGaps() is run. Also returns derivatives w.r.t. displacement and pressure.
  auto forces = formulation.computeNodalForces( integ_rule, nodal_gaps );

  return forces;
}

// Interface classes

namespace tribol {

struct ElementPairRule {
  std::pair<int, int> pair;
  std::vector<std::array<double, 3>> points_1;
  std::vector<std::array<double, 3>> points_2;
  std::vector<double> weights;
};

struct IntegrationRule {
  std::vector<ElementPairRule> pair_rules;
};

using PairList = std::vector<std::pair<int, int>>;

class IntegrationRuleGenerator {
 public:
  virtual ~IntegrationRuleGenerator() = 0;
  virtual PairList updateSearch() {};
  virtual IntegrationRule createIntegrationRule( [[maybe_unused]] std::optional<PairList> pairs ) {};

 protected:
  virtual void updateFields() {};
};

inline IntegrationRuleGenerator::~IntegrationRuleGenerator() = default;

/**
 * @brief Generic integration rule defined through template arguments
 *
 * @tparam _FieldData Holds field data. 2 main options for this template argument: MfemFieldData or TribolFieldData.
 * TribolFieldData has Tribol-local data (serial) and updateTribolData() is a no-op. MfemFieldData has MFEM data (MPI
 * parallel) and Tribol-local data (serial) and updateTribolData() transfers the latest MFEM fields to their analogous
 * Tribol fields. This is a template parameter, rather than standard runtime OOP, because the types of the fields are
 * different based on the template argument.
 * @tparam _Search Template arguments are classes that identify a list of element pairs (PairList) through a findPairs()
 * method. Generally, there is a 2 step approach: CoarseSearch step and a PairFilter step.
 * @tparam _PairRule Given an element pair, find an integration rule (ElementPairRule).
 */
template <typename _FieldData, typename _Search, typename _PairRule>
class IntegrationMethod : public IntegrationRuleGenerator {
 public:
  using FieldDataT = _FieldData;
  using SearchT = _Search;
  using PairRuleT = _PairRule;
  IntegrationMethod( FieldDataT&& field_data, SearchT&& search, PairRuleT&& pair_rule )
      : field_data_( field_data ), search_( search ), pair_rule_( pair_rule )
  {
  }

  PairList updateSearch() override
  {
    updateFields();
    return search_.findPairs( field_data_ );
  }

  IntegrationRule createIntegrationRule( std::optional<PairList> pairs = std::nullopt )
  {
    if ( !pairs ) {
      pairs = updateSearch();
    }

    // allocate ElementPairRules in IntegrationRule

    // loop over PairList, call pair_rule.createIntegrationRule( pair );

    // cleanup; return IntegrationRule
  }

 private:
  void updateFields() override { field_data_.updateTribolData(); }

  FieldDataT field_data_;
  SearchT search_;
  PairRuleT pair_rule_;
};

// TODO: Gap evaluator interface class + generic implementation; contact formulation interface class + generic
// implementation

}  // namespace tribol
