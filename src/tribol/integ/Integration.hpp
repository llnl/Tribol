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

/// Selector for the triangle quadrature family used by GaussPolyIntTri().
enum TriangleQuadratureRuleFamily
{
  TRI_RULE_LEGACY,
  TRI_RULE_SYMMETRIC
};

/// Maximum number of quadrature points in the built-in symmetric triangle rules.
constexpr int max_symmetric_triangle_qpts = 25;
/// Maximum number of quadrature points in the built-in Gauss-Legendre segment rules.
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
 * \param [in] k order of integration
 * \param [in] family selector for the triangle quadrature family
 *
 * \pre order 2 <= k <= 10
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */
void GaussPolyIntTri( SurfaceContactElem const& elem, IntegPts& integ, int k,
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

TRIBOL_HOST_DEVICE inline void GetCommonPlaneOverlapCentroid( SurfaceContactElem const& elem, RealT cx[3] )
{
  cx[0] = 0.;
  cx[1] = 0.;
  cx[2] = 0.;

  if ( elem.dim == 2 ) {
    VertexAvgCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert, cx[0], cx[1], cx[2] );
  } else {
    PolyAreaCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert, cx[0], cx[1], cx[2] );
  }
}

TRIBOL_HOST_DEVICE inline void AccumulateCommonPlaneIntegralAtPoint( SurfaceContactElem const& elem, const RealT x[3],
                                                                     const RealT wt, RealT* const integ1,
                                                                     RealT* const integ2 )
{
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    RealT phi1 = 0.;
    RealT phi2 = 0.;
    EvalBasisOnPhysicalFace( elem.faceCoords1, x[0], x[1], x[2], elem.numFaceVert, a, phi1 );
    EvalBasisOnPhysicalFace( elem.faceCoords2, x[0], x[1], x[2], elem.numFaceVert, a, phi2 );
    integ1[a] += wt * phi1;
    integ2[a] += wt * phi2;
  }
}

/*!
 * \brief Returns the legacy triangle quadrature rule historically used by
 *        CommonPlane and GaussPolyIntTri().
 *
 * \param [in] order requested rule order
 * \param [out] wts quadrature weights normalized so they sum to 1 on a triangle
 * \param [out] coords quadrature coordinates stored as stacked (xi, eta) pairs
 *
 * \note This legacy rule is only available for the previously supported
 *       orders 2 and 3/4 and is kept for regression comparison tests.
 */
TRIBOL_HOST_DEVICE inline int GetLegacyTriangleRule( int order, RealT* wts, RealT* coords )
{
  switch ( order ) {
    case 2:
      wts[0] = 0.3333333333;
      wts[1] = 0.3333333333;
      wts[2] = 0.3333333333;

      coords[0] = 0.1666666667;
      coords[1] = 0.1666666667;
      coords[2] = 0.6666666667;
      coords[3] = 0.1666666667;
      coords[4] = 0.1666666667;
      coords[5] = 0.6666666667;
      return 3;
    case 3:
    case 4: {
      constexpr RealT wt1 = 0.109951743655322;
      constexpr RealT wt2 = 0.223381589678011;
      wts[0] = wt1;
      wts[1] = wt1;
      wts[2] = wt1;
      wts[3] = wt2;
      wts[4] = wt2;
      wts[5] = wt2;

      constexpr RealT x1 = 0.091576213509771;
      constexpr RealT x2 = 0.816847572980459;
      constexpr RealT x3 = 0.108103018168070;
      constexpr RealT x4 = 0.445948490915965;
      coords[0] = x1;
      coords[1] = x1;
      coords[2] = x2;
      coords[3] = x1;
      coords[4] = x1;
      coords[5] = x2;
      coords[6] = x3;
      coords[7] = x4;
      coords[8] = x4;
      coords[9] = x3;
      coords[10] = x4;
      coords[11] = x4;
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
struct SymmetricTriangleRuleData
{
  int num_centroid_orbits;
  int num_edge_orbits;
  int num_general_orbits;
  const RealT* weights;
  const RealT* orbits;
};

constexpr RealT symmetric_triangle_weight_scale = 0.5;

constexpr RealT tri_deg2_weights[] = { 6.66666666666666666666666666666666635e-01 };
constexpr RealT tri_deg2_orbits[] = { 1.66666666666666666666666666666666659e-01, 6.66666666666666666666666666666666635e-01 };

constexpr RealT tri_deg4_weights[] = { 4.46763179356022931390014016866245598e-01, 2.19903487310643735276652649800421061e-01 };
constexpr RealT tri_deg4_orbits[] = { 4.45948490915964886318329253883051984e-01, 1.08103018168070227363341492233896033e-01,
                                      9.15762135097707434595714634022014804e-02, 8.16847572980458513080857073195597039e-01 };

constexpr RealT tri_deg5_weights[] = { 4.50000000000000000000000000000000010e-01, 2.51878361089654305191367891000362687e-01,
                                       2.64788305577012361475298775666303977e-01 };
constexpr RealT tri_deg5_orbits[] = { 3.33333333333333333333333333333333317e-01, 1.01286507323456338800987361915123836e-01,
                                      7.97426985353087322398025276169752328e-01, 4.70142064105115089770441209513447613e-01,
                                      5.97158717897698204591175809731048219e-02 };

constexpr RealT tri_deg6_weights[] = { 1.01689812740413633841873618213737963e-01, 2.33572551452758732050579222771158894e-01,
                                       1.65702151236747150387106912840884901e-01 };
constexpr RealT tri_deg6_orbits[] = { 6.30890144915022283403316028708191300e-02, 8.73821971016995543319336794258361644e-01,
                                      2.49286745170910421291638553107019076e-01, 5.01426509658179157416722893785961848e-01,
                                      5.31450498448169473532496716313981651e-02, 6.36502499121398647230142594412049640e-01,
                                      3.10352451033784405416607733956552146e-01 };

constexpr RealT tri_deg7_weights[] = { 3.30901002215842620719558969458348911e-02, 2.55888342460311145565802470369292636e-01,
                                       1.54173292371972135669643041667482776e-01, 1.11757465806399561679632628842028190e-01 };
constexpr RealT tri_deg7_orbits[] = { 3.37306485545878487149717263008162317e-02, 9.32538702890824302570056547398367537e-01,
                                      2.41577382595403558950186769837781999e-01, 5.16845234809192882099626460324436002e-01,
                                      4.74309692504718234209580735949185780e-01, 5.13806149905635315808385281016284391e-02,
                                      4.70366446525952333414099753568849895e-02, 7.54280040550053177356239324628119970e-01,
                                      1.98683314797351589302350700014995040e-01 };

constexpr RealT tri_deg8_weights[] = { 2.88631215355574336502182220978129237e-01, 1.90183268534569249587792208777168633e-01,
                                       2.06434741069436500563583100584258068e-01, 6.49169952463961606218518566835611904e-02,
                                       5.44606283488699885296893801478178481e-02 };
constexpr RealT tri_deg8_orbits[] = { 3.33333333333333333333333333333333317e-01, 4.59292588292723156028815514494169350e-01,
                                      8.14148234145536879423689710116613481e-02, 1.70569307751760206622293501491464506e-01,
                                      6.58861384496479586755412997017070988e-01, 5.05472283170309754584235505965989197e-02,
                                      8.98905543365938049083152898806802161e-01, 8.39477740995760533721383453929445768e-03,
                                      7.28492392955404281241000379176061966e-01, 2.63112829634638113421785786284643576e-01 };

constexpr RealT tri_deg9_weights[] = { 1.94271592565597667638483965014577269e-01, 1.55655082009548558633478712598807923e-01,
                                       1.59295477854420506065783548528090548e-01, 6.26694004542781410737096625744186273e-02,
                                       5.11553513173960625233575971179996460e-02, 8.65670787545787545787545787545787526e-02 };
constexpr RealT tri_deg9_orbits[] = { 3.33333333333333333333333333333333317e-01, 4.37089591492936637269930364435354971e-01,
                                      1.25820817014126725460139271129290058e-01, 1.88203535619032730240961280467335542e-01,
                                      6.23592928761934539518077439065328819e-01, 4.89682519198737627783706924836192818e-01,
                                      2.06349616025247444325861503276144129e-02, 4.47295133944527098651065899662763588e-02,
                                      9.10540973211094580269786820067447282e-01, 3.68384120547362836348175987833851049e-02,
                                      7.41198598784498020690079873523423793e-01, 2.21962989160765695675102527693191078e-01 };

constexpr RealT tri_deg10_weights[] = { 1.63486658292571932856237369968355216e-01, 2.67059376262991325511459567981373070e-02,
                                        9.19159272094894560275758192650956353e-02, 1.27809812792848090865797467525306648e-01,
                                        6.83692963259188572573831680826845816e-02, 5.05955154145767687780855813656664345e-02 };
constexpr RealT tri_deg10_orbits[] = { 3.33333333333333333333333333333333317e-01, 3.20553732169435129309845893364897379e-02,
                                       9.35889253566112974138030821327020524e-01, 1.42161101056564385092162103190958311e-01,
                                       7.15677797886871229815675793618083377e-01, 3.21812995288835421225097560986048687e-01,
                                       5.30054118927344028277095673945694069e-01, 1.48132885783820550497806765068257172e-01,
                                       2.96198894887297676338362694260427776e-02, 6.01233328683459245454742893458687815e-01,
                                       3.69146781827810986911420837115269408e-01, 2.83676653399384392504357555781301898e-02,
                                       8.07930600922879065079949902881744115e-01, 1.63701733737182495669614341540125695e-01 };

TRIBOL_HOST_DEVICE inline bool GetSymmetricTriangleRuleData( int order, SymmetricTriangleRuleData& rule )
{
  switch ( order ) {
    case 2:
      rule = { 0, 1, 0, tri_deg2_weights, tri_deg2_orbits };
      return true;
    case 3:
    case 4:
      rule = { 0, 2, 0, tri_deg4_weights, tri_deg4_orbits };
      return true;
    case 5:
      rule = { 1, 2, 0, tri_deg5_weights, tri_deg5_orbits };
      return true;
    case 6:
      rule = { 0, 2, 1, tri_deg6_weights, tri_deg6_orbits };
      return true;
    case 7:
      rule = { 0, 3, 1, tri_deg7_weights, tri_deg7_orbits };
      return true;
    case 8:
      rule = { 1, 3, 1, tri_deg8_weights, tri_deg8_orbits };
      return true;
    case 9:
      rule = { 1, 4, 1, tri_deg9_weights, tri_deg9_orbits };
      return true;
    case 10:
      rule = { 1, 2, 3, tri_deg10_weights, tri_deg10_orbits };
      return true;
    default:
      return false;
  }
}

TRIBOL_HOST_DEVICE inline int ExpandSymmetricTriangleRule( const SymmetricTriangleRuleData& rule, RealT* wts, RealT* coords )
{
  int w_idx = 0;
  int c_idx = 0;
  int q_idx = 0;

  for ( int orbit = 0; orbit < rule.num_centroid_orbits; ++orbit ) {
    const RealT a = rule.orbits[c_idx++];
    wts[q_idx] = symmetric_triangle_weight_scale * rule.weights[w_idx++];
    coords[2 * q_idx] = a;
    coords[2 * q_idx + 1] = a;
    ++q_idx;
  }

  for ( int orbit = 0; orbit < rule.num_edge_orbits; ++orbit ) {
    const RealT a = rule.orbits[c_idx++];
    const RealT b = rule.orbits[c_idx++];
    const RealT w = symmetric_triangle_weight_scale * rule.weights[w_idx++];

    wts[q_idx] = w;
    coords[2 * q_idx] = a;
    coords[2 * q_idx + 1] = b;
    ++q_idx;

    wts[q_idx] = w;
    coords[2 * q_idx] = b;
    coords[2 * q_idx + 1] = a;
    ++q_idx;

    wts[q_idx] = w;
    coords[2 * q_idx] = a;
    coords[2 * q_idx + 1] = a;
    ++q_idx;
  }

  for ( int orbit = 0; orbit < rule.num_general_orbits; ++orbit ) {
    const RealT a = rule.orbits[c_idx++];
    const RealT b = rule.orbits[c_idx++];
    const RealT c = rule.orbits[c_idx++];
    const RealT w = symmetric_triangle_weight_scale * rule.weights[w_idx++];

    const RealT xi_eta[6][2] = { { b, c }, { c, b }, { a, c }, { c, a }, { a, b }, { b, a } };
    for ( int i = 0; i < 6; ++i ) {
      wts[q_idx] = w;
      coords[2 * q_idx] = xi_eta[i][0];
      coords[2 * q_idx + 1] = xi_eta[i][1];
      ++q_idx;
    }
  }

  return q_idx;
}

}  // namespace detail

/*!
 * \brief Returns the built-in higher-order symmetric triangle quadrature rule.
 *
 * \param [in] order requested rule order
 * \param [out] wts quadrature weights normalized so they sum to 1 on a triangle
 * \param [out] coords quadrature coordinates stored as stacked (xi, eta) pairs
 *
 * \note Orders 2 through 10 are available. Order 3 uses the same minimal rule
 *       as order 4, matching the PETSc/Witherden-Vincent data set.
 */
TRIBOL_HOST_DEVICE inline int GetCommonPlaneTriangleRule( int order, RealT* wts, RealT* coords )
{
  detail::SymmetricTriangleRuleData rule;
  if ( !detail::GetSymmetricTriangleRuleData( order, rule ) ) {
#ifdef TRIBOL_USE_HOST
    SLIC_ERROR( "GetCommonPlaneTriangleRule(): only symmetric triangle integration of order 2-10 is implemented." );
#endif
    return 0;
  }
  return detail::ExpandSymmetricTriangleRule( rule, wts, coords );
}

/*!
 * \brief Returns the requested triangle quadrature rule family.
 *
 * \param [in] order requested rule order
 * \param [in] family selector for legacy versus symmetric rule data
 * \param [out] wts quadrature weights normalized so they sum to 1 on a triangle
 * \param [out] coords quadrature coordinates stored as stacked (xi, eta) pairs
 */
TRIBOL_HOST_DEVICE inline int GetTriangleRule( int order, TriangleQuadratureRuleFamily family, RealT* wts, RealT* coords )
{
  switch ( family ) {
    case TRI_RULE_LEGACY:
      return GetLegacyTriangleRule( order, wts, coords );
    case TRI_RULE_SYMMETRIC:
      return GetCommonPlaneTriangleRule( order, wts, coords );
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
      coords[1] = 1.29234424311301331233889510925404835e-01;
      coords[2] = 2.97483866778698624490873198198455490e-01;
      coords[3] = 5.00000000000000000000000000000000000e-01;
      coords[4] = 7.02516133221301375509126801801544510e-01;
      coords[5] = 8.70765575688698668766110489074595165e-01;
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

TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegralCommonPlaneMultiPoint( SurfaceContactElem const& elem,
                                                                          const int quadrature_order,
                                                                          RealT* const integ1, RealT* const integ2 )
{
  if ( elem.dim == 2 ) {
    RealT rule_wts[max_segment_gauss_legendre_qpts] = { 0. };
    RealT rule_coords[max_segment_gauss_legendre_qpts] = { 0. };
    const int num_qpts = GetCommonPlaneSegmentRule( quadrature_order, rule_wts, rule_coords );

    const RealT x0 = elem.overlapCoords[0];
    const RealT y0 = elem.overlapCoords[1];
    const RealT x1 = elem.overlapCoords[2];
    const RealT y1 = elem.overlapCoords[3];
    const RealT length = magnitude( x1 - x0, y1 - y0 );

    for ( int qp = 0; qp < num_qpts; ++qp ) {
      const RealT s = rule_coords[qp];
      const RealT one_minus_s = 1. - s;
      RealT x[3] = { one_minus_s * x0 + s * x1, one_minus_s * y0 + s * y1, 0. };
      AccumulateCommonPlaneIntegralAtPoint( elem, x, length * rule_wts[qp], integ1, integ2 );
    }
    return;
  }

  constexpr int max_qpts = max_symmetric_triangle_qpts;
  RealT rule_wts[max_qpts] = { 0. };
  RealT rule_coords[2 * max_qpts] = { 0. };
  const int num_qpts = GetCommonPlaneTriangleRule( quadrature_order, rule_wts, rule_coords );

  RealT centroid[3];
  GetCommonPlaneOverlapCentroid( elem, centroid );

  RealT xTri[3];
  RealT yTri[3];
  RealT zTri[3];

  for ( int j = 0; j < elem.numPolyVert; ++j ) {
    const int next = ( j == elem.numPolyVert - 1 ) ? 0 : j + 1;
    xTri[0] = elem.overlapCoords[elem.dim * j];
    yTri[0] = elem.overlapCoords[elem.dim * j + 1];
    zTri[0] = elem.overlapCoords[elem.dim * j + 2];
    xTri[1] = elem.overlapCoords[elem.dim * next];
    yTri[1] = elem.overlapCoords[elem.dim * next + 1];
    zTri[1] = elem.overlapCoords[elem.dim * next + 2];
    xTri[2] = centroid[0];
    yTri[2] = centroid[1];
    zTri[2] = centroid[2];

    const RealT area = Area3DTri( xTri, yTri, zTri );
    if ( area <= 0. ) {
      continue;
    }

    for ( int qp = 0; qp < num_qpts; ++qp ) {
      const RealT xi = rule_coords[2 * qp];
      const RealT eta = rule_coords[2 * qp + 1];
      const RealT n0 = 1. - xi - eta;
      RealT x[3];
      x[0] = n0 * xTri[0] + xi * xTri[1] + eta * xTri[2];
      x[1] = n0 * yTri[0] + xi * yTri[1] + eta * yTri[2];
      x[2] = n0 * zTri[0] + xi * zTri[1] + eta * zTri[2];
      AccumulateCommonPlaneIntegralAtPoint( elem, x, area * rule_wts[qp], integ1, integ2 );
    }
  }
}

TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegralCommonPlane( SurfaceContactElem const& elem, const PolyInteg rule,
                                                                const int quadrature_order, RealT* const integ1,
                                                                RealT* const integ2 )
{
  switch ( rule ) {
    case SINGLE_POINT: {
      RealT cx[3] = { 0., 0., 0. };
      GetCommonPlaneOverlapCentroid( elem, cx );
      AccumulateCommonPlaneIntegralAtPoint( elem, cx, 1.0, integ1, integ2 );
      break;
    }
    case MULTI_POINT:
      EvalWeakFormIntegralCommonPlaneMultiPoint( elem, quadrature_order, integ1, integ2 );
      break;
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "EvalWeakFormIntegralCommonPlane(): unsupported polygon integration rule." );
#endif
      break;
  }
}

template <>
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegral<COMMON_PLANE, SINGLE_POINT>( SurfaceContactElem const& elem,
                                                                                 RealT* const integ1,
                                                                                 RealT* const integ2 )
{
  RealT cx[3] = { 0., 0., 0. };
  GetCommonPlaneOverlapCentroid( elem, cx );
  AccumulateCommonPlaneIntegralAtPoint( elem, cx, 1.0, integ1, integ2 );
}

}  // end namespace tribol
#endif /* SRC_TRIBOL_INTEG_INTEGRATION_HPP_ */
