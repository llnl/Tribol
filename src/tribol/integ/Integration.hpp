// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTEGRATION_HPP_
#define SRC_TRIBOL_INTEG_INTEGRATION_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

namespace tribol {

// forward declaration
struct SurfaceContactElem;

/// struct to hold 2D or 3D integration point coordinates and
//  weights for integration on a face-face overlapping
//  convex polygon. This struct is quadrature rule agnostic.
struct IntegPts {
  /// IntegPts constructor
  IntegPts( int numPoints,  ///< [in] Number of integration points
            int IPDim       ///< [in] dimension of integration point coordinates
            )
      : numIPs( numPoints ), ipDim( IPDim )
  {
    xy = new RealT[IPDim * numPoints];
    wts = new RealT[numPoints];
  }

  /// IntegPts overloaded constructor
  IntegPts() : numIPs( 0 ), xy( nullptr ), wts( nullptr ) {}

  /// Destructor
  ~IntegPts()
  {
    if ( xy != nullptr ) {
      delete[] xy;
      xy = nullptr;
    }
    if ( wts != nullptr ) {
      delete[] wts;
      wts = nullptr;
    }
  }

  /// Initialization function
  void initialize( int const dim, int const numTotalIPs )
  {
    this->ipDim = dim;
    this->numIPs = numTotalIPs;
    if ( this->xy == nullptr ) {
      this->xy = new RealT[dim * numTotalIPs];
    } else {
      delete[] this->xy;
      this->xy = new RealT[dim * numTotalIPs];
    }
    if ( this->wts == nullptr ) {
      this->wts = new RealT[numTotalIPs];
    } else {
      delete[] this->wts;
      this->wts = new RealT[numTotalIPs];
    }
  }

  // member variables
  int numIPs;  ///< number of integration points on entire overlap
  int ipDim;   ///< coordinate dimension of the integration points
  RealT* xy;   ///< coordinates of ALL integration points
  RealT* wts;  ///< integration point weights
};

/*!
 *
 * \brief Templated function with explicit specialization evaluating the
 *        weak form contact integral, typically involving the integration
 *        of shape functions or product of shape functions over contact
 *        overlap patches for surface-to-surface contact methods.
 *
 * \param [in] elem surface contact element struct
 * \param [out] integ1 scalar integral evaluation for face 1 at node nodeEvalId
 * \param [out] integ2 scalar integral evaluation for face 2 at node nodeEvalId
 *
 * \pre The local node id, nodeEvalId, ranges from 0-3 for a four node quad face.
 *
 */
template <ContactMethod M, PolyInteg I>
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegral( SurfaceContactElem const& elem, RealT* const integ1,
                                                     RealT* const integ2 );

/** @brief Selector for the triangle quadrature family used by GaussPolyIntTri(). */
enum TriangleQuadratureRuleFamily
{
  TRI_RULE_LEGACY,    ///< Historical Tribol rules used by mortar integration.
  TRI_RULE_SYMMETRIC  ///< Symmetric rules used by multipoint CommonPlane integration.
};

/** Maximum number of quadrature points in the built-in symmetric triangle rules. */
constexpr int max_symmetric_triangle_qpts = 25;
/** Maximum number of quadrature points in the built-in Gauss-Legendre segment rules. */
constexpr int max_segment_gauss_legendre_qpts = 10;

/*!
 *
 * \brief Populates the integration points and weights on the IntegPts object
 *        for all integration points per Taylor-Wingate-Bos integration rule
 *        of order k.
 *
 * \note Integration per M. Taylor, B. Wingate, L. Bos. Several new quadrature
 *       formulas for polynomial integration in the triangle.
 *       arXiv:math/0501496, 2007.
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in,out] integ IntegPts object holding integration points and weights
 * \param [in] k order of TWB integration
 *
 * \pre order 2 <= k <= 3
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */
void TWBPolyInt( SurfaceContactElem const& elem, IntegPts& integ, int k );

/*!
 *
 * \brief Populates the integration points and weights on the IntegPts object
 *        for all integration points per symmetric Gauss integration rule
 *        of order k on triangles
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in,out] integ IntegPts object holding integration points and weights
 * \param [in] order order of integration
 * \param [in] family selector for the triangle quadrature family
 *
 * \pre order 2 <= k <= 10
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */
void GaussPolyIntTri( SurfaceContactElem const& elem, IntegPts& integ, int order,
                      TriangleQuadratureRuleFamily family = TRI_RULE_SYMMETRIC );

/*!
 *
 * \brief Populates the integration points and weights on the IntegPts object
 *        for all integration points per symmetric Gauss integration rule
 *        of order k on quadrilaterals
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in,out] integ IntegPts object holding integration points and weights
 * \param [in] k order of integration
 *
 * \pre order 2 <= k <= 3
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */

void GaussPolyIntQuad( SurfaceContactElem const& elem, IntegPts& integ, int k );
/*!
 *
 * \brief returns the number of TWB integration points for polygonal overlap
 *        for integration rule of order k
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in] k order of TWB integration
 *
 * \pre order 2 <= k <= 3
 *
 */
int NumTWBPointsPoly( SurfaceContactElem const& elem, int k );

/*!
 *
 * \brief returns the number of TWB integration points on a triangle per
 *        the integration rule of order k
 *
 * \param [in] order order of polynomial that TWB integration rule will exactly integrate
 *
 * \pre order 2 <= k <= 3
 *
 */
int NumTWBPointsPerTri( int order );

//-----------------------------------------------------------------------------
// Implementations
//-----------------------------------------------------------------------------

/**
 * @brief Compute the centroid used to decompose a CommonPlane overlap cell.
 *
 * @param elem Contact element containing the overlap interval or polygon
 * @param overlap_centroid Output physical overlap centroid
 */
TRIBOL_HOST_DEVICE inline void GetCommonPlaneOverlapCentroid( SurfaceContactElem const& elem,
                                                              RealT overlap_centroid[3] )
{
  overlap_centroid[0] = 0.;
  overlap_centroid[1] = 0.;
  overlap_centroid[2] = 0.;

  if ( elem.dim == 2 ) {
    VertexAvgCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert, overlap_centroid[0], overlap_centroid[1],
                       overlap_centroid[2] );
  } else {
    PolyAreaCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert, overlap_centroid[0], overlap_centroid[1],
                      overlap_centroid[2] );
  }
}

/**
 * @brief Accumulate both face-basis integrals at one physical overlap point.
 *
 * @param elem Contact element containing both face coordinate fields
 * @param integration_point Physical CommonPlane integration point
 * @param integration_weight Physical integration weight at the point
 * @param first_face_integrals Accumulated first-face basis integrals
 * @param second_face_integrals Accumulated second-face basis integrals
 */
TRIBOL_HOST_DEVICE inline void AccumulateCommonPlaneIntegralAtPoint( SurfaceContactElem const& elem,
                                                                     const RealT integration_point[3],
                                                                     const RealT integration_weight,
                                                                     RealT* const first_face_integrals,
                                                                     RealT* const second_face_integrals )
{
  for ( int basis_index = 0; basis_index < elem.numFaceVert; ++basis_index ) {
    RealT first_basis_value = 0.;
    RealT second_basis_value = 0.;
    EvalBasisOnPhysicalFace( elem.faceCoords1, integration_point[0], integration_point[1], integration_point[2],
                             elem.numFaceVert, basis_index, first_basis_value );
    EvalBasisOnPhysicalFace( elem.faceCoords2, integration_point[0], integration_point[1], integration_point[2],
                             elem.numFaceVert, basis_index, second_basis_value );
    first_face_integrals[basis_index] += integration_weight * first_basis_value;
    second_face_integrals[basis_index] += integration_weight * second_basis_value;
  }
}

/*!
 * \brief Returns the legacy triangle quadrature rule historically used by
 *        CommonPlane and GaussPolyIntTri().
 *
 * \param [in] order requested rule order
 * \param [out] quadrature_weights quadrature weights normalized so they sum to 1 on a triangle
 * \param [out] reference_coordinates quadrature coordinates stored as stacked (xi, eta) pairs
 * \return number of quadrature points in the selected rule
 *
 * \note This legacy rule is only available for the previously supported
 *       orders 2 and 3/4 and is kept for regression comparison tests.
 */
TRIBOL_HOST_DEVICE inline int GetLegacyTriangleRule( int order, RealT* quadrature_weights,
                                                     RealT* reference_coordinates )
{
  switch ( order ) {
    case 2:
      quadrature_weights[0] = 0.3333333333;
      quadrature_weights[1] = 0.3333333333;
      quadrature_weights[2] = 0.3333333333;

      reference_coordinates[0] = 0.1666666667;
      reference_coordinates[1] = 0.1666666667;
      reference_coordinates[2] = 0.6666666667;
      reference_coordinates[3] = 0.1666666667;
      reference_coordinates[4] = 0.1666666667;
      reference_coordinates[5] = 0.6666666667;
      return 3;
    case 3:
    case 4: {
      constexpr RealT first_weight = 0.109951743655322;
      constexpr RealT second_weight = 0.223381589678011;
      quadrature_weights[0] = first_weight;
      quadrature_weights[1] = first_weight;
      quadrature_weights[2] = first_weight;
      quadrature_weights[3] = second_weight;
      quadrature_weights[4] = second_weight;
      quadrature_weights[5] = second_weight;

      constexpr RealT first_orbit_coordinate = 0.091576213509771;
      constexpr RealT second_orbit_coordinate = 0.816847572980459;
      constexpr RealT third_orbit_coordinate = 0.108103018168070;
      constexpr RealT fourth_orbit_coordinate = 0.445948490915965;
      reference_coordinates[0] = first_orbit_coordinate;
      reference_coordinates[1] = first_orbit_coordinate;
      reference_coordinates[2] = second_orbit_coordinate;
      reference_coordinates[3] = first_orbit_coordinate;
      reference_coordinates[4] = first_orbit_coordinate;
      reference_coordinates[5] = second_orbit_coordinate;
      reference_coordinates[6] = third_orbit_coordinate;
      reference_coordinates[7] = fourth_orbit_coordinate;
      reference_coordinates[8] = fourth_orbit_coordinate;
      reference_coordinates[9] = third_orbit_coordinate;
      reference_coordinates[10] = fourth_orbit_coordinate;
      reference_coordinates[11] = fourth_orbit_coordinate;
      return 6;
    }
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "GetLegacyTriangleRule(): only legacy Gauss integration of order 2-4 is implemented." );
#endif
      return 0;
  }
}

namespace detail {

/*!
 * \brief Minimal symmetric triangle quadrature orbit data.
 *
 * \note The compact orbit tables below are adapted from the symmetric triangle
 *       rules distributed in PETSc, which cite
 *       F.D. Witherden and P.E. Vincent,
 *       "On the identification of symmetric quadrature rules for finite element methods",
 *       Computers & Mathematics with Applications 69(10), 2015,
 *       doi:10.1016/j.camwa.2015.03.017.
 *
 *       PETSc stores weights for a reference triangle of area 2. Tribol uses
 *       weights normalized so the weights sum to 1 and the physical triangle
 *       area is applied separately, so the imported weights are scaled by 1/2
 *       during expansion.
 */
struct SymmetricTriangleRuleData {
  /** Number of one-point centroid orbits. */
  int num_centroid_orbits;

  /** Number of three-point orbits with two equal barycentric coordinates. */
  int num_edge_orbits;

  /** Number of six-point orbits with three distinct barycentric coordinates. */
  int num_general_orbits;

  /** Maximum number of distinct orbits in a supported rule. */
  static constexpr int maximum_orbit_count{ 6 };

  /** Maximum number of compact barycentric coordinates in a supported rule. */
  static constexpr int maximum_orbit_coordinate_count{ 14 };

  /** One quadrature weight for each orbit. */
  RealT weights[maximum_orbit_count]{};

  /** Independent barycentric coordinates for each orbit. */
  RealT orbits[maximum_orbit_coordinate_count]{};
};

/** Scale from the source rule's area-two triangle to Tribol's unit-sum convention. */
constexpr RealT symmetric_triangle_weight_scale = 0.5;

/**
 * @brief Select compact symmetric triangle quadrature data for an order.
 *
 * @param order Requested polynomial integration order
 * @param rule Selected compact orbit data
 * @return true when the requested order is supported
 */
TRIBOL_HOST_DEVICE inline bool GetSymmetricTriangleRuleData( int order, SymmetricTriangleRuleData& rule )
{
  switch ( order ) {
    case 2:
      rule = { 0,
               1,
               0,
               { 6.66666666666666666666666666666666635e-01 },
               { 1.66666666666666666666666666666666659e-01, 6.66666666666666666666666666666666635e-01 } };
      return true;
    case 3:
    case 4:
      rule = { 0,
               2,
               0,
               { 4.46763179356022931390014016866245598e-01, 2.19903487310643735276652649800421061e-01 },
               { 4.45948490915964886318329253883051984e-01, 1.08103018168070227363341492233896033e-01,
                 9.15762135097707434595714634022014804e-02, 8.16847572980458513080857073195597039e-01 } };
      return true;
    case 5:
      rule = { 1,
               2,
               0,
               { 4.50000000000000000000000000000000010e-01, 2.51878361089654305191367891000362687e-01,
                 2.64788305577012361475298775666303977e-01 },
               { 3.33333333333333333333333333333333317e-01, 1.01286507323456338800987361915123836e-01,
                 7.97426985353087322398025276169752328e-01, 4.70142064105115089770441209513447613e-01,
                 5.97158717897698204591175809731048219e-02 } };
      return true;
    case 6:
      rule = { 0,
               2,
               1,
               { 1.01689812740413633841873618213737963e-01, 2.33572551452758732050579222771158894e-01,
                 1.65702151236747150387106912840884901e-01 },
               { 6.30890144915022283403316028708191300e-02, 8.73821971016995543319336794258361644e-01,
                 2.49286745170910421291638553107019076e-01, 5.01426509658179157416722893785961848e-01,
                 5.31450498448169473532496716313981651e-02, 6.36502499121398647230142594412049640e-01,
                 3.10352451033784405416607733956552146e-01 } };
      return true;
    case 7:
      rule = { 0,
               3,
               1,
               { 3.30901002215842620719558969458348911e-02, 2.55888342460311145565802470369292636e-01,
                 1.54173292371972135669643041667482776e-01, 1.11757465806399561679632628842028190e-01 },
               { 3.37306485545878487149717263008162317e-02, 9.32538702890824302570056547398367537e-01,
                 2.41577382595403558950186769837781999e-01, 5.16845234809192882099626460324436002e-01,
                 4.74309692504718234209580735949185780e-01, 5.13806149905635315808385281016284391e-02,
                 4.70366446525952333414099753568849895e-02, 7.54280040550053177356239324628119970e-01,
                 1.98683314797351589302350700014995040e-01 } };
      return true;
    case 8:
      rule = { 1,
               3,
               1,
               { 2.88631215355574336502182220978129237e-01, 1.90183268534569249587792208777168633e-01,
                 2.06434741069436500563583100584258068e-01, 6.49169952463961606218518566835611904e-02,
                 5.44606283488699885296893801478178481e-02 },
               { 3.33333333333333333333333333333333317e-01, 4.59292588292723156028815514494169350e-01,
                 8.14148234145536879423689710116613481e-02, 1.70569307751760206622293501491464506e-01,
                 6.58861384496479586755412997017070988e-01, 5.05472283170309754584235505965989197e-02,
                 8.98905543365938049083152898806802161e-01, 8.39477740995760533721383453929445768e-03,
                 7.28492392955404281241000379176061966e-01, 2.63112829634638113421785786284643576e-01 } };
      return true;
    case 9:
      rule = { 1,
               4,
               1,
               { 1.94271592565597667638483965014577269e-01, 1.55655082009548558633478712598807923e-01,
                 1.59295477854420506065783548528090548e-01, 6.26694004542781410737096625744186273e-02,
                 5.11553513173960625233575971179996460e-02, 8.65670787545787545787545787545787526e-02 },
               { 3.33333333333333333333333333333333317e-01, 4.37089591492936637269930364435354971e-01,
                 1.25820817014126725460139271129290058e-01, 1.88203535619032730240961280467335542e-01,
                 6.23592928761934539518077439065328819e-01, 4.89682519198737627783706924836192818e-01,
                 2.06349616025247444325861503276144129e-02, 4.47295133944527098651065899662763588e-02,
                 9.10540973211094580269786820067447282e-01, 3.68384120547362836348175987833851049e-02,
                 7.41198598784498020690079873523423793e-01, 2.21962989160765695675102527693191078e-01 } };
      return true;
    case 10:
      rule = { 1,
               2,
               3,
               { 1.63486658292571932856237369968355216e-01, 2.67059376262991325511459567981373070e-02,
                 9.19159272094894560275758192650956353e-02, 1.27809812792848090865797467525306648e-01,
                 6.83692963259188572573831680826845816e-02, 5.05955154145767687780855813656664345e-02 },
               { 3.33333333333333333333333333333333317e-01, 3.20553732169435129309845893364897379e-02,
                 9.35889253566112974138030821327020524e-01, 1.42161101056564385092162103190958311e-01,
                 7.15677797886871229815675793618083377e-01, 3.21812995288835421225097560986048687e-01,
                 5.30054118927344028277095673945694069e-01, 1.48132885783820550497806765068257172e-01,
                 2.96198894887297676338362694260427776e-02, 6.01233328683459245454742893458687815e-01,
                 3.69146781827810986911420837115269408e-01, 2.83676653399384392504357555781301898e-02,
                 8.07930600922879065079949902881744115e-01, 1.63701733737182495669614341540125695e-01 } };
      return true;
    default:
      return false;
  }
}

/**
 * @brief Expand compact symmetric triangle orbits into quadrature points.
 *
 * @param rule Compact symmetric triangle quadrature data
 * @param quadrature_weights Expanded quadrature weights
 * @param reference_coordinates Expanded stacked reference coordinates
 * @return Number of expanded quadrature points
 */
TRIBOL_HOST_DEVICE inline int ExpandSymmetricTriangleRule( const SymmetricTriangleRuleData& rule,
                                                           RealT* quadrature_weights, RealT* reference_coordinates )
{
  int weight_index = 0;
  int orbit_coordinate_index = 0;
  int quadrature_point_index = 0;

  for ( int orbit = 0; orbit < rule.num_centroid_orbits; ++orbit ) {
    const RealT centroid_coordinate = rule.orbits[orbit_coordinate_index++];
    quadrature_weights[quadrature_point_index] = symmetric_triangle_weight_scale * rule.weights[weight_index++];
    reference_coordinates[2 * quadrature_point_index] = centroid_coordinate;
    reference_coordinates[2 * quadrature_point_index + 1] = centroid_coordinate;
    ++quadrature_point_index;
  }

  for ( int orbit = 0; orbit < rule.num_edge_orbits; ++orbit ) {
    const RealT repeated_coordinate = rule.orbits[orbit_coordinate_index++];
    const RealT distinct_coordinate = rule.orbits[orbit_coordinate_index++];
    const RealT quadrature_weight = symmetric_triangle_weight_scale * rule.weights[weight_index++];

    quadrature_weights[quadrature_point_index] = quadrature_weight;
    reference_coordinates[2 * quadrature_point_index] = repeated_coordinate;
    reference_coordinates[2 * quadrature_point_index + 1] = distinct_coordinate;
    ++quadrature_point_index;

    quadrature_weights[quadrature_point_index] = quadrature_weight;
    reference_coordinates[2 * quadrature_point_index] = distinct_coordinate;
    reference_coordinates[2 * quadrature_point_index + 1] = repeated_coordinate;
    ++quadrature_point_index;

    quadrature_weights[quadrature_point_index] = quadrature_weight;
    reference_coordinates[2 * quadrature_point_index] = repeated_coordinate;
    reference_coordinates[2 * quadrature_point_index + 1] = repeated_coordinate;
    ++quadrature_point_index;
  }

  for ( int orbit = 0; orbit < rule.num_general_orbits; ++orbit ) {
    const RealT first_coordinate = rule.orbits[orbit_coordinate_index++];
    const RealT second_coordinate = rule.orbits[orbit_coordinate_index++];
    const RealT third_coordinate = rule.orbits[orbit_coordinate_index++];
    const RealT quadrature_weight = symmetric_triangle_weight_scale * rule.weights[weight_index++];

    const RealT coordinate_permutations[6][2] = {
        { second_coordinate, third_coordinate }, { third_coordinate, second_coordinate },
        { first_coordinate, third_coordinate },  { third_coordinate, first_coordinate },
        { first_coordinate, second_coordinate }, { second_coordinate, first_coordinate } };
    for ( int permutation_index = 0; permutation_index < 6; ++permutation_index ) {
      quadrature_weights[quadrature_point_index] = quadrature_weight;
      reference_coordinates[2 * quadrature_point_index] = coordinate_permutations[permutation_index][0];
      reference_coordinates[2 * quadrature_point_index + 1] = coordinate_permutations[permutation_index][1];
      ++quadrature_point_index;
    }
  }

  return quadrature_point_index;
}

}  // namespace detail

/*!
 * \brief Returns the built-in higher-order symmetric triangle quadrature rule.
 *
 * \param [in] order requested rule order
 * \param [out] quadrature_weights quadrature weights normalized so they sum to 1 on a triangle
 * \param [out] reference_coordinates quadrature coordinates stored as stacked (xi, eta) pairs
 * \return number of quadrature points in the selected rule
 *
 * \note Orders 2 through 10 are available. Order 3 uses the same minimal rule
 *       as order 4, matching the PETSc/Witherden-Vincent data set.
 */
TRIBOL_HOST_DEVICE inline int GetCommonPlaneTriangleRule( int order, RealT* quadrature_weights,
                                                          RealT* reference_coordinates )
{
  detail::SymmetricTriangleRuleData rule;
  if ( !detail::GetSymmetricTriangleRuleData( order, rule ) ) {
#ifdef TRIBOL_USE_HOST
    SLIC_ERROR( "GetCommonPlaneTriangleRule(): only symmetric triangle integration of order 2-10 is implemented." );
#endif
    return 0;
  }
  return detail::ExpandSymmetricTriangleRule( rule, quadrature_weights, reference_coordinates );
}

/*!
 * \brief Returns the requested triangle quadrature rule family.
 *
 * \param [in] order requested rule order
 * \param [in] family selector for legacy versus symmetric rule data
 * \param [out] quadrature_weights quadrature weights normalized so they sum to 1 on a triangle
 * \param [out] reference_coordinates quadrature coordinates stored as stacked (xi, eta) pairs
 * \return number of quadrature points in the selected rule
 */
TRIBOL_HOST_DEVICE inline int GetTriangleRule( int order, TriangleQuadratureRuleFamily family,
                                               RealT* quadrature_weights, RealT* reference_coordinates )
{
  switch ( family ) {
    case TRI_RULE_LEGACY:
      return GetLegacyTriangleRule( order, quadrature_weights, reference_coordinates );
    case TRI_RULE_SYMMETRIC:
      return GetCommonPlaneTriangleRule( order, quadrature_weights, reference_coordinates );
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "GetTriangleRule(): unsupported triangle rule family." );
#endif
      return 0;
  }
}

/*!
 * \brief Returns a Gauss-Legendre quadrature rule on the unit segment.
 *
 * \param [in] order requested rule order
 * \param [out] wts quadrature weights normalized so they sum to 1 on [0,1]
 * \param [out] coords quadrature coordinates on [0,1]
 * \return number of quadrature points in the selected rule
 */
TRIBOL_HOST_DEVICE inline int GetCommonPlaneSegmentRule( int order, RealT* wts, RealT* coords )
{
  switch ( order ) {
    case 2:
      wts[0] = 5.00000000000000000000000000000000000e-01;
      wts[1] = 5.00000000000000000000000000000000000e-01;
      coords[0] = 2.11324865405187117745425609795414482e-01;
      coords[1] = 7.88675134594812882254574390204585518e-01;
      return 2;
    case 3:
      wts[0] = 2.77777777777777777777777777777777778e-01;
      wts[1] = 4.44444444444444444444444444444444444e-01;
      wts[2] = 2.77777777777777777777777777777777778e-01;
      coords[0] = 1.12701665379258311482063373571554511e-01;
      coords[1] = 5.00000000000000000000000000000000000e-01;
      coords[2] = 8.87298334620741688517936626428445489e-01;
      return 3;
    case 4:
      wts[0] = 1.73927422568726928648300228976864863e-01;
      wts[1] = 3.26072577431273071351699771023135137e-01;
      wts[2] = 3.26072577431273071351699771023135137e-01;
      wts[3] = 1.73927422568726928648300228976864863e-01;
      coords[0] = 6.94318442029737123880253935661376383e-02;
      coords[1] = 3.30009478207571867549864872986354601e-01;
      coords[2] = 6.69990521792428132450135127013645399e-01;
      coords[3] = 9.30568155797026287611974606433862362e-01;
      return 4;
    case 5:
      wts[0] = 1.18463442528094543757132020373224693e-01;
      wts[1] = 2.39314335249683234020645713783081311e-01;
      wts[2] = 2.84444444444444444444444444444444444e-01;
      wts[3] = 2.39314335249683234020645713783081311e-01;
      wts[4] = 1.18463442528094543757132020373224693e-01;
      coords[0] = 4.69100770306680036011865699692305193e-02;
      coords[1] = 2.30765344947158454446500534703347781e-01;
      coords[2] = 5.00000000000000000000000000000000000e-01;
      coords[3] = 7.69234655052841545553499465296652219e-01;
      coords[4] = 9.53089922969331996398813430030769481e-01;
      return 5;
    case 6:
      wts[0] = 8.56622461895851725230519499689802902e-02;
      wts[1] = 1.80380786524069303841036681987673464e-01;
      wts[2] = 2.33956967286345520487813812579846245e-01;
      wts[3] = 2.33956967286345520487813812579846245e-01;
      wts[4] = 1.80380786524069303841036681987673464e-01;
      wts[5] = 8.56622461895851725230519499689802902e-02;
      coords[0] = 3.37652428984239962556928330124159244e-02;
      coords[1] = 1.69395306766867745483983092697870979e-01;
      coords[2] = 3.80690406958401560316832671838599160e-01;
      coords[3] = 6.19309593041598439683167328161400840e-01;
      coords[4] = 8.30604693233132254516016907302129021e-01;
      coords[5] = 9.66234757101576003744307166987584076e-01;
      return 6;
    case 7:
      wts[0] = 6.47424830844348466391116955198397926e-02;
      wts[1] = 1.39852695744638333950704650054195659e-01;
      wts[2] = 1.90915025252559472475161990594647293e-01;
      wts[3] = 2.08979591836734693877551020408163265e-01;
      wts[4] = 1.90915025252559472475161990594647293e-01;
      wts[5] = 1.39852695744638333950704650054195659e-01;
      wts[6] = 6.47424830844348466391116955198397926e-02;
      coords[0] = 2.54460438286207377369030550090367355e-02;
      coords[1] = 1.29234407200302780068067613359605864e-01;
      coords[2] = 2.97077424311301416546967632731488033e-01;
      coords[3] = 5.00000000000000000000000000000000000e-01;
      coords[4] = 7.02922575688698583453032367268511967e-01;
      coords[5] = 8.70765592799697219931932386640394136e-01;
      coords[6] = 9.74553956171379262263096944990963264e-01;
      return 7;
    case 8:
      wts[0] = 5.06142681451881693180916895511138059e-02;
      wts[1] = 1.11190517226687235272177997268125811e-01;
      wts[2] = 1.56853322938943643668981100993387281e-01;
      wts[3] = 1.81341891689181001927177820586651362e-01;
      wts[4] = 1.81341891689181001927177820586651362e-01;
      wts[5] = 1.56853322938943643668981100993387281e-01;
      wts[6] = 1.11190517226687235272177997268125811e-01;
      wts[7] = 5.06142681451881693180916895511138059e-02;
      coords[0] = 1.98550717512318841525047110753824461e-02;
      coords[1] = 1.01666761293186647733518599687717188e-01;
      coords[2] = 2.37233795041835507091130475405343431e-01;
      coords[3] = 4.08282678752175097530261928819908057e-01;
      coords[4] = 5.91717321247824902469738071180091943e-01;
      coords[5] = 7.62766204958164492908869524594656569e-01;
      coords[6] = 8.98333238706813352266481400312282812e-01;
      coords[7] = 9.80144928248768115847495288924617554e-01;
      return 8;
    case 9:
      wts[0] = 4.06371941807872005172919364720240956e-02;
      wts[1] = 9.03240803474287173109739194798907182e-02;
      wts[2] = 1.30305348201467649015160147969872934e-01;
      wts[3] = 1.56173538520001468666934369973026078e-01;
      wts[4] = 1.65119677500629881523396871610248955e-01;
      wts[5] = 1.56173538520001468666934369973026078e-01;
      wts[6] = 1.30305348201467649015160147969872934e-01;
      wts[7] = 9.03240803474287173109739194798907182e-02;
      wts[8] = 4.06371941807872005172919364720240956e-02;
      coords[0] = 1.59198802461869216995971690127085065e-02;
      coords[1] = 8.19844463366820870961097794728460945e-02;
      coords[2] = 1.93314283649704707966974877456187964e-01;
      coords[3] = 3.37873288298095542617664572568897352e-01;
      coords[4] = 5.00000000000000000000000000000000000e-01;
      coords[5] = 6.62126711701904457382335427431102648e-01;
      coords[6] = 8.06685716350295292033025122543812036e-01;
      coords[7] = 9.18015553663317912903890220527153906e-01;
      coords[8] = 9.84080119753813078300402830987291494e-01;
      return 9;
    case 10:
      wts[0] = 3.33356721543440461444643439469389906e-02;
      wts[1] = 7.47256745752902979112342760845433925e-02;
      wts[2] = 1.09543181257990906041775930711667835e-01;
      wts[3] = 1.34633359654998153565044736633326588e-01;
      wts[4] = 1.47762112357376435165896479362247582e-01;
      wts[5] = 1.47762112357376435165896479362247582e-01;
      wts[6] = 1.34633359654998153565044736633326588e-01;
      wts[7] = 1.09543181257990906041775930711667835e-01;
      wts[8] = 7.47256745752902979112342760845433925e-02;
      wts[9] = 3.33356721543440461444643439469389906e-02;
      coords[0] = 1.30467357414141598997194531116613714e-02;
      coords[1] = 6.74683166555077431721327998540848508e-02;
      coords[2] = 1.60295215850487803884800815437504493e-01;
      coords[3] = 2.83302302935376372885100587561369794e-01;
      coords[4] = 4.25562830509184389676645012342252051e-01;
      coords[5] = 5.74437169490815610323354987657747949e-01;
      coords[6] = 7.16697697064623627114899412438630206e-01;
      coords[7] = 8.39704784149512196115199184562495507e-01;
      coords[8] = 9.32531683344492256827867200145915149e-01;
      coords[9] = 9.86953264258585840100280546888338629e-01;
      return 10;
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "GetCommonPlaneSegmentRule(): only Gauss-Legendre integration of order 2-10 is implemented." );
#endif
      return 0;
  }
}

/**
 * @brief Integrate both CommonPlane face bases with a multipoint rule.
 *
 * Two-dimensional overlap intervals use Gauss-Legendre quadrature. Three-dimensional
 * overlap polygons are decomposed into a nonoverlapping fan about their centroid,
 * and every fan triangle uses the requested symmetric triangle rule.
 *
 * @param elem Contact element containing the overlap interval or polygon
 * @param quadrature_order Requested integration order in the supported range [2,10]
 * @param first_face_integrals Accumulated first-face basis integrals
 * @param second_face_integrals Accumulated second-face basis integrals
 */
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegralCommonPlaneMultiPoint( SurfaceContactElem const& elem,
                                                                          const int quadrature_order,
                                                                          RealT* const first_face_integrals,
                                                                          RealT* const second_face_integrals )
{
  if ( elem.dim == 2 ) {
    RealT quadrature_weights[max_segment_gauss_legendre_qpts] = { 0. };
    RealT reference_coordinates[max_segment_gauss_legendre_qpts] = { 0. };
    const int number_of_quadrature_points =
        GetCommonPlaneSegmentRule( quadrature_order, quadrature_weights, reference_coordinates );

    const RealT first_endpoint_x = elem.overlapCoords[0];
    const RealT first_endpoint_y = elem.overlapCoords[1];
    const RealT second_endpoint_x = elem.overlapCoords[2];
    const RealT second_endpoint_y = elem.overlapCoords[3];
    const RealT overlap_length =
        magnitude( second_endpoint_x - first_endpoint_x, second_endpoint_y - first_endpoint_y );

    for ( int quadrature_point = 0; quadrature_point < number_of_quadrature_points; ++quadrature_point ) {
      const RealT segment_coordinate = reference_coordinates[quadrature_point];
      const RealT first_endpoint_weight = 1. - segment_coordinate;
      RealT integration_point[3] = { first_endpoint_weight * first_endpoint_x + segment_coordinate * second_endpoint_x,
                                     first_endpoint_weight * first_endpoint_y + segment_coordinate * second_endpoint_y,
                                     0. };
      AccumulateCommonPlaneIntegralAtPoint( elem, integration_point,
                                            overlap_length * quadrature_weights[quadrature_point], first_face_integrals,
                                            second_face_integrals );
    }
    return;
  }

  RealT quadrature_weights[max_symmetric_triangle_qpts] = { 0. };
  RealT reference_coordinates[2 * max_symmetric_triangle_qpts] = { 0. };
  const int number_of_quadrature_points =
      GetCommonPlaneTriangleRule( quadrature_order, quadrature_weights, reference_coordinates );

  RealT overlap_centroid[3];
  GetCommonPlaneOverlapCentroid( elem, overlap_centroid );

  RealT triangle_x_coordinates[3];
  RealT triangle_y_coordinates[3];
  RealT triangle_z_coordinates[3];

  for ( int overlap_vertex = 0; overlap_vertex < elem.numPolyVert; ++overlap_vertex ) {
    const int next_overlap_vertex = overlap_vertex == elem.numPolyVert - 1 ? 0 : overlap_vertex + 1;
    triangle_x_coordinates[0] = elem.overlapCoords[elem.dim * overlap_vertex];
    triangle_y_coordinates[0] = elem.overlapCoords[elem.dim * overlap_vertex + 1];
    triangle_z_coordinates[0] = elem.overlapCoords[elem.dim * overlap_vertex + 2];
    triangle_x_coordinates[1] = elem.overlapCoords[elem.dim * next_overlap_vertex];
    triangle_y_coordinates[1] = elem.overlapCoords[elem.dim * next_overlap_vertex + 1];
    triangle_z_coordinates[1] = elem.overlapCoords[elem.dim * next_overlap_vertex + 2];
    triangle_x_coordinates[2] = overlap_centroid[0];
    triangle_y_coordinates[2] = overlap_centroid[1];
    triangle_z_coordinates[2] = overlap_centroid[2];

    const RealT triangle_area = Area3DTri( triangle_x_coordinates, triangle_y_coordinates, triangle_z_coordinates );
    if ( triangle_area <= 0. ) {
      continue;
    }

    for ( int quadrature_point = 0; quadrature_point < number_of_quadrature_points; ++quadrature_point ) {
      const RealT first_triangle_coordinate = reference_coordinates[2 * quadrature_point];
      const RealT second_triangle_coordinate = reference_coordinates[2 * quadrature_point + 1];
      const RealT third_triangle_coordinate = 1. - first_triangle_coordinate - second_triangle_coordinate;
      RealT integration_point[3];
      integration_point[0] = third_triangle_coordinate * triangle_x_coordinates[0] +
                             first_triangle_coordinate * triangle_x_coordinates[1] +
                             second_triangle_coordinate * triangle_x_coordinates[2];
      integration_point[1] = third_triangle_coordinate * triangle_y_coordinates[0] +
                             first_triangle_coordinate * triangle_y_coordinates[1] +
                             second_triangle_coordinate * triangle_y_coordinates[2];
      integration_point[2] = third_triangle_coordinate * triangle_z_coordinates[0] +
                             first_triangle_coordinate * triangle_z_coordinates[1] +
                             second_triangle_coordinate * triangle_z_coordinates[2];
      AccumulateCommonPlaneIntegralAtPoint( elem, integration_point,
                                            triangle_area * quadrature_weights[quadrature_point], first_face_integrals,
                                            second_face_integrals );
    }
  }
}

/**
 * @brief Integrate both CommonPlane face bases with the selected overlap rule.
 *
 * @param elem Contact element containing the overlap interval or polygon
 * @param rule CommonPlane overlap integration rule
 * @param quadrature_order Requested multipoint integration order
 * @param first_face_integrals Accumulated first-face basis integrals
 * @param second_face_integrals Accumulated second-face basis integrals
 */
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegralCommonPlane( SurfaceContactElem const& elem, const PolyInteg rule,
                                                                const int quadrature_order,
                                                                RealT* const first_face_integrals,
                                                                RealT* const second_face_integrals )
{
  switch ( rule ) {
    case SINGLE_POINT: {
      RealT overlap_centroid[3] = { 0., 0., 0. };
      GetCommonPlaneOverlapCentroid( elem, overlap_centroid );
      AccumulateCommonPlaneIntegralAtPoint( elem, overlap_centroid, 1.0, first_face_integrals, second_face_integrals );
      break;
    }
    case MULTI_POINT:
      EvalWeakFormIntegralCommonPlaneMultiPoint( elem, quadrature_order, first_face_integrals, second_face_integrals );
      break;
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "EvalWeakFormIntegralCommonPlane(): unsupported polygon integration rule." );
#endif
      break;
  }
}

/**
 * @brief Evaluate legacy one-point CommonPlane weak-form basis integrals.
 *
 * @param elem Contact element containing the overlap geometry and face coordinates
 * @param integ1 First-face basis integral values
 * @param integ2 Second-face basis integral values
 */
template <>
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegral<COMMON_PLANE, SINGLE_POINT>( SurfaceContactElem const& elem,
                                                                                 RealT* const integ1,
                                                                                 RealT* const integ2 )
{
  RealT overlap_centroid[3] = { 0., 0., 0. };
  GetCommonPlaneOverlapCentroid( elem, overlap_centroid );
  AccumulateCommonPlaneIntegralAtPoint( elem, overlap_centroid, 1.0, integ1, integ2 );
}

}  // end namespace tribol
#endif /* SRC_TRIBOL_INTEG_INTEGRATION_HPP_ */
