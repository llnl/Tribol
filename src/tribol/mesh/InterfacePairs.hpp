// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_MESH_INTERFACE_PAIRS_HPP_
#define SRC_MESH_INTERFACE_PAIRS_HPP_

#include "tribol/types.hpp"
#include "tribol/common/Parameters.hpp"

#include "axom/slic.hpp"

namespace tribol {

struct InterfacePair {
  InterfacePair( integer m1, integer t1, integer i1, integer m2, integer t2, integer i2 )
      : pairId( -1 ),
        meshId1( m1 ),
        pairType1( t1 ),
        pairIndex1( i1 ),
        meshId2( m2 ),
        pairType2( t2 ),
        pairIndex2( i2 ),
        isContactCandidate( true )
  {
  }

  // overload constructor with pair id
  InterfacePair( integer m1, integer t1, integer i1, integer m2, integer t2, integer i2, integer id )
      : pairId( id ),
        meshId1( m1 ),
        pairType1( t1 ),
        pairIndex1( i1 ),
        meshId2( m2 ),
        pairType2( t2 ),
        pairIndex2( i2 ),
        isContactCandidate( true )
  {
  }

  // overload constructor with boolean indicating contact candidacy
  InterfacePair( integer m1, integer t1, integer i1, integer m2, integer t2, integer i2, bool isCandidate, integer id )
      : pairId( id ),
        meshId1( m1 ),
        pairType1( t1 ),
        pairIndex1( i1 ),
        meshId2( m2 ),
        pairType2( t2 ),
        pairIndex2( i2 ),
        isContactCandidate( isCandidate )
  {
  }

  // overload constructor to handle zero input arguments
  InterfacePair()
      : pairId( -1 ),
        meshId1( -1 ),
        pairType1( -1 ),
        pairIndex1( -1 ),
        meshId2( -1 ),
        pairType2( -1 ),
        pairIndex2( -1 ),
        isContactCandidate( true )
  {
  }

  // pair id
  integer pairId;

  // mesh id, face type, and face id for face 1
  integer meshId1;
  integer pairType1;
  integer pairIndex1;

  // mesh id, face type and face id for face 2
  integer meshId2;
  integer pairType2;
  integer pairIndex2;

  // boolean indicating if a binned pair is a contact candidate.
  // A contact candidate is defined as a face-pair that is deemed geometrically proximate
  // by the binning coarse search, and one that passes the finer computational geometry
  // (CG) filter/checks. These finer checks identify face-pairs that are intersecting or
  // nearly intersecting with positive areas of overlap. These checks do not indicate
  // whether a face-pair is contacting, since the definition of 'contacting' is specific
  // to a particular contact method and its enforced contstraints. Rather, the CG filter
  // identifies contact candidacy, or face-pairs likely in contact.
  bool isContactCandidate;
};

class InterfacePairs {
 public:
  InterfacePairs() = default;
  ~InterfacePairs() = default;

  void clear();
  void reserve( integer capacity );

  void addInterfacePair( InterfacePair const& pair );

  void updateInterfacePair( InterfacePair const& pair, integer const idx );

  void setMeshId( int side, int id )
  {
    SLIC_ERROR_IF( side != 1 && side != 2, "mesh side not 1 or 2." );
    if ( side == 1 ) m_meshId1 = id;
    if ( side == 2 ) m_meshId2 = id;
  }

  void setPairType( int side, integer type )
  {
    SLIC_ERROR_IF( side != 1 && side != 2, "mesh side not 1 or 2." );
    if ( side == 1 ) m_pairType1 = type;
    if ( side == 2 ) m_pairType2 = type;
  }

  InterfacePair getInterfacePair( IndexType idx ) const;

  IndexType getNumPairs() const { return static_cast<IndexType>( m_pairIndex1.size() ); }

 private:
  // A list of interface pairs is defined to be between
  // two meshes with the same elements type on each mesh
  integer m_meshId1;
  integer m_meshId2;
  integer m_pairType1;
  integer m_pairType2;

  containerArray<integer> m_pairIndex1;
  containerArray<integer> m_pairIndex2;
  containerArray<bool> m_isContactCandidate;
};

} /* namespace tribol */

#endif /* SRC_MESH_INTERFACE_PAIRS_HPP_ */
