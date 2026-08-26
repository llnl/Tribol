// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_FIELDDATA_HPP_
#define SRC_TRIBOL_PHYSICS_FIELDDATA_HPP_

#include "tribol/config.hpp"

#include "tribol/common/BasicTypes.hpp"

#include <atomic>
#include <cstdint>

namespace tribol {

enum class FieldDataBackend
{
  Tribol,
  Mfem
};

class FieldDataBase {
 public:
  explicit FieldDataBase( FieldDataBackend backend );
  virtual ~FieldDataBase() = default;

  FieldDataBase( const FieldDataBase& ) = delete;
  FieldDataBase& operator=( const FieldDataBase& ) = delete;
  FieldDataBase( FieldDataBase&& ) = delete;
  FieldDataBase& operator=( FieldDataBase&& ) = delete;

  FieldDataBackend backend() const { return backend_; }
  std::uint64_t instanceId() const { return instance_id_; }
  std::uint64_t fieldGeneration() const { return field_generation_; }
  std::uint64_t topologyGeneration() const { return topology_generation_; }

 protected:
  void markFieldsUpdated() { ++field_generation_; }
  void markTopologyUpdated()
  {
    ++topology_generation_;
    ++field_generation_;
  }

 private:
  static std::atomic<std::uint64_t> next_instance_id_;

  FieldDataBackend backend_;
  std::uint64_t instance_id_;
  std::uint64_t field_generation_{ 0 };
  std::uint64_t topology_generation_{ 0 };
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_FIELDDATA_HPP_ */
