// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ELEMENTFORCEANDGAPENZYME_HPP_
#define SRC_TRIBOL_PHYSICS_ELEMENTFORCEANDGAPENZYME_HPP_

#include "tribol/physics/ElementForceAndGap.hpp"

#ifdef TRIBOL_USE_ENZYME
#include "tribol/common/Enzyme.hpp"
#endif

namespace tribol {

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Class that implements ElementForceAndGap using Enzyme for derivatives.
 *
 * @tparam ForceModel The force model policy (e.g., MortarElementForce).
 * @tparam Dim The spatial dimension.
 */
template <typename ForceModel, int Dim>
class ElementForceAndGapEnzyme : public ElementForceAndGap<ForceModel, Dim> {
 public:
  void computeGap( const InterfacePair& pair, const IntegrationRule<ForceModel, Dim>& integration_rule,
                   ForceModel& gap_method, const std::vector<RealT>& coords1, const std::vector<RealT>& coords2,
                   mfem::Vector& gaps, mfem::DenseMatrix& dg_dx1, mfem::DenseMatrix& dg_dx2 ) override
  {
    // coords1 = Mortar (Tribol Mesh 1) -> x2
    // coords2 = Nonmortar (Tribol Mesh 2) -> x1
    const int size1 = coords2.size() / Dim; // Nonmortar size
    const int size2 = coords1.size() / Dim; // Mortar size

    const RealT* x1 = coords2.data();
    const RealT* x2 = coords1.data();
    const RealT* n1 = gap_method.n1.data();
    
    // We don't have pressure for gap computation, assume 0 or dummy
    std::vector<RealT> p1(size1, 0.0);
    
    // Outputs
    std::vector<RealT> f1(size1 * Dim);
    std::vector<RealT> f2(size2 * Dim);
    std::vector<RealT> g1(size1);

    // Resize gap method outputs
    gap_method.dg1_dn1.SetSize(size1, size1 * Dim);
    gap_method.dg1_dn1 = 0.0;

    // Enzyme seeds
    std::vector<RealT> x1_dot(size1 * Dim, 0.0);
    std::vector<RealT> x2_dot(size2 * Dim, 0.0);
    std::vector<RealT> n1_dot(size1 * Dim, 0.0);

    // Compute dg/dx1 (x1 = Nonmortar = coords2)
    // Note: dg_dx2 in args is w.r.t coords2 (Mesh 2, Nonmortar) -> dg_dx1 in mortar notation
    dg_dx2.SetSize(size1, size1 * Dim); // Mesh 2
    for (int i = 0; i < size1 * Dim; ++i) {
        x1_dot[i] = 1.0;
        std::vector<RealT> dg1_val(size1, 0.0); // derivative output
        std::vector<RealT> df1_dummy(size1 * Dim);
        std::vector<RealT> df2_dummy(size2 * Dim);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_DUP, x1, x1_dot.data(),
            TRIBOL_ENZYME_CONST, n1,
            TRIBOL_ENZYME_CONST, p1.data(),
            TRIBOL_ENZYME_DUP, f1.data(), df1_dummy.data(),
            TRIBOL_ENZYME_DUP, g1.data(), dg1_val.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_CONST, x2,
            TRIBOL_ENZYME_DUP, f2.data(), df2_dummy.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1; ++j) {
            dg_dx2(j, i) = dg1_val[j];
        }
        x1_dot[i] = 0.0;
    }

    // Compute dg/dx2 (x2 = Mortar = coords1)
    // Note: dg_dx1 in args is w.r.t coords1 (Mesh 1, Mortar) -> dg_dx2 in mortar notation
    dg_dx1.SetSize(size1, size2 * Dim); // Mesh 1
    for (int i = 0; i < size2 * Dim; ++i) {
        x2_dot[i] = 1.0;
        std::vector<RealT> dg1_val(size1, 0.0);
        std::vector<RealT> df1_dummy(size1 * Dim);
        std::vector<RealT> df2_dummy(size2 * Dim);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_CONST, x1,
            TRIBOL_ENZYME_CONST, n1,
            TRIBOL_ENZYME_CONST, p1.data(),
            TRIBOL_ENZYME_DUP, f1.data(), df1_dummy.data(),
            TRIBOL_ENZYME_DUP, g1.data(), dg1_val.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_DUP, x2, x2_dot.data(),
            TRIBOL_ENZYME_DUP, f2.data(), df2_dummy.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1; ++j) {
            dg_dx1(j, i) = dg1_val[j];
        }
        x2_dot[i] = 0.0;
    }

    // Compute dg/dn1
    for (int i = 0; i < size1 * Dim; ++i) {
        n1_dot[i] = 1.0;
        std::vector<RealT> dg1_val(size1, 0.0);
        std::vector<RealT> df1_dummy(size1 * Dim);
        std::vector<RealT> df2_dummy(size2 * Dim);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_CONST, x1,
            TRIBOL_ENZYME_DUP, n1, n1_dot.data(),
            TRIBOL_ENZYME_CONST, p1.data(),
            TRIBOL_ENZYME_DUP, f1.data(), df1_dummy.data(),
            TRIBOL_ENZYME_DUP, g1.data(), dg1_val.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_CONST, x2,
            TRIBOL_ENZYME_DUP, f2.data(), df2_dummy.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1; ++j) {
            gap_method.dg1_dn1(j, i) = dg1_val[j];
        }
        n1_dot[i] = 0.0;
    }
    
    // Compute values (re-run or use last run? Enzyme usually computes primal too, but with const inputs it's cleaner to separate or reuse)
    // We need just the primal values for gaps
    ForceModel::compute(x1, n1, p1.data(), f1.data(), g1.data(), size1, x2, f2.data(), size2);
    gaps.SetSize(size1);
    for(int i=0; i<size1; ++i) gaps(i) = g1[i];
  }

  void computeGap( const InterfacePair& pair, const IntegrationRule<ForceModel, Dim>& integration_rule,
                   ForceModel& gap_method, const std::vector<RealT>& coords1, const std::vector<RealT>& coords2,
                   mfem::Vector& gaps ) override
  {
    const int size1 = coords2.size() / Dim; // Nonmortar
    const int size2 = coords1.size() / Dim; // Mortar
    const RealT* x1 = coords2.data();
    const RealT* x2 = coords1.data();
    const RealT* n1 = gap_method.n1.data();
    std::vector<RealT> p1(size1, 0.0);
    std::vector<RealT> f1(size1 * Dim);
    std::vector<RealT> f2(size2 * Dim);
    std::vector<RealT> g1(size1);

    ForceModel::compute(x1, n1, p1.data(), f1.data(), g1.data(), size1, x2, f2.data(), size2);
    
    gaps.SetSize(size1);
    for(int i=0; i<size1; ++i) gaps(i) = g1[i];
  }

  void computeForce( const InterfacePair& pair, const IntegrationRule<ForceModel, Dim>& integration_rule,
                     ForceModel& gap_method, const std::vector<RealT>& coords1, const std::vector<RealT>& coords2,
                     std::shared_ptr<ArrayViewT<RealT>> pressures, mfem::Vector& f1, mfem::Vector& f2,
                     mfem::DenseMatrix& df1_dx1, mfem::DenseMatrix& df1_dx2, mfem::DenseMatrix& df2_dx1,
                     mfem::DenseMatrix& df2_dx2, mfem::DenseMatrix& df1_dp, mfem::DenseMatrix& df2_dp,
                     RealT& energy ) override
  {
    // coords1 = Mortar (Mesh 1) -> x2
    // coords2 = Nonmortar (Mesh 2) -> x1
    // f1 (output) = Mesh 1 (Mortar) -> maps from local f2
    // f2 (output) = Mesh 2 (Nonmortar) -> maps from local f1
    
    const int size1 = coords2.size() / Dim; // Nonmortar
    const int size2 = coords1.size() / Dim; // Mortar

    const RealT* x1 = coords2.data();
    const RealT* x2 = coords1.data();
    const RealT* n1 = gap_method.n1.data();
    const RealT* p1 = (pressures && pressures->size() > 0) ? pressures->data() : nullptr;
    
    std::vector<RealT> p1_dummy(size1, 0.0);
    if (!p1) p1 = p1_dummy.data();

    // Outputs
    std::vector<RealT> local_f1(size1 * Dim);
    std::vector<RealT> local_f2(size2 * Dim);
    std::vector<RealT> g1(size1);

    // Resize extra derivatives
    gap_method.df1_dn1.SetSize(size1 * Dim, size1 * Dim);
    gap_method.df2_dn1.SetSize(size2 * Dim, size1 * Dim);
    gap_method.dg1_dn1.SetSize(size1, size1 * Dim); // if needed here too
    
    gap_method.df1_dn1 = 0.0;
    gap_method.df2_dn1 = 0.0;
    gap_method.dg1_dn1 = 0.0;

    // Seeds
    std::vector<RealT> x1_dot(size1 * Dim, 0.0);
    std::vector<RealT> x2_dot(size2 * Dim, 0.0);
    std::vector<RealT> n1_dot(size1 * Dim, 0.0);
    std::vector<RealT> p1_dot(size1, 0.0);

    // 1. Differentiate w.r.t x1 (Nonmortar, Mesh 2)
    // Corresponds to df2_dx2 and df1_dx2 in ElementForceAndGap notation
    // (Mesh 2 -> f2 output, Mesh 1 -> f1 output)
    // local_f1 is force on x1 (Mesh 2). local_f2 is force on x2 (Mesh 1).
    // So local_f1 derivatives -> df2_...
    //    local_f2 derivatives -> df1_...
    df2_dx2.SetSize(size1 * Dim, size1 * Dim);
    df1_dx2.SetSize(size2 * Dim, size1 * Dim);

    for (int i = 0; i < size1 * Dim; ++i) {
        x1_dot[i] = 1.0;
        std::vector<RealT> df1_val(size1 * Dim);
        std::vector<RealT> df2_val(size2 * Dim);
        std::vector<RealT> dg1_val(size1);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_DUP, x1, x1_dot.data(),
            TRIBOL_ENZYME_CONST, n1,
            TRIBOL_ENZYME_CONST, p1,
            TRIBOL_ENZYME_DUP, local_f1.data(), df1_val.data(),
            TRIBOL_ENZYME_DUP, g1.data(), dg1_val.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_CONST, x2,
            TRIBOL_ENZYME_DUP, local_f2.data(), df2_val.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1 * Dim; ++j) df2_dx2(j, i) = df1_val[j];
        for (int j = 0; j < size2 * Dim; ++j) df1_dx2(j, i) = df2_val[j];
        x1_dot[i] = 0.0;
    }

    // 2. Differentiate w.r.t x2 (Mortar, Mesh 1)
    // Corresponds to df2_dx1 and df1_dx1
    df2_dx1.SetSize(size1 * Dim, size2 * Dim);
    df1_dx1.SetSize(size2 * Dim, size2 * Dim);

    for (int i = 0; i < size2 * Dim; ++i) {
        x2_dot[i] = 1.0;
        std::vector<RealT> df1_val(size1 * Dim);
        std::vector<RealT> df2_val(size2 * Dim);
        std::vector<RealT> dg1_val(size1);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_CONST, x1,
            TRIBOL_ENZYME_CONST, n1,
            TRIBOL_ENZYME_CONST, p1,
            TRIBOL_ENZYME_DUP, local_f1.data(), df1_val.data(),
            TRIBOL_ENZYME_DUP, g1.data(), dg1_val.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_DUP, x2, x2_dot.data(),
            TRIBOL_ENZYME_DUP, local_f2.data(), df2_val.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1 * Dim; ++j) df2_dx1(j, i) = df1_val[j];
        for (int j = 0; j < size2 * Dim; ++j) df1_dx1(j, i) = df2_val[j];
        x2_dot[i] = 0.0;
    }

    // 3. Differentiate w.r.t n1 (Normals)
    for (int i = 0; i < size1 * Dim; ++i) {
        n1_dot[i] = 1.0;
        std::vector<RealT> df1_val(size1 * Dim);
        std::vector<RealT> df2_val(size2 * Dim);
        std::vector<RealT> dg1_val(size1);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_CONST, x1,
            TRIBOL_ENZYME_DUP, n1, n1_dot.data(),
            TRIBOL_ENZYME_CONST, p1,
            TRIBOL_ENZYME_DUP, local_f1.data(), df1_val.data(),
            TRIBOL_ENZYME_DUP, g1.data(), dg1_val.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_CONST, x2,
            TRIBOL_ENZYME_DUP, local_f2.data(), df2_val.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1 * Dim; ++j) gap_method.df1_dn1(j, i) = df1_val[j];
        for (int j = 0; j < size2 * Dim; ++j) gap_method.df2_dn1(j, i) = df2_val[j];
        n1_dot[i] = 0.0;
    }

    // 4. Differentiate w.r.t p1 (Pressure)
    // df1_dp corresponds to df2_dp (Mesh 2 force w.r.t pressure)
    // df2_dp corresponds to df1_dp (Mesh 1 force w.r.t pressure)
    df2_dp.SetSize(size1 * Dim, size1);
    df1_dp.SetSize(size2 * Dim, size1);

    for (int i = 0; i < size1; ++i) {
        p1_dot[i] = 1.0;
        std::vector<RealT> df1_val(size1 * Dim);
        std::vector<RealT> df2_val(size2 * Dim);
        std::vector<RealT> dg1_val(size1);

        __enzyme_fwddiff<void>((void*)ForceModel::compute,
            TRIBOL_ENZYME_CONST, x1,
            TRIBOL_ENZYME_CONST, n1,
            TRIBOL_ENZYME_DUP, p1, p1_dot.data(),
            TRIBOL_ENZYME_DUP, local_f1.data(), df1_val.data(),
            TRIBOL_ENZYME_CONST, g1.data(),
            TRIBOL_ENZYME_CONST, size1,
            TRIBOL_ENZYME_CONST, x2,
            TRIBOL_ENZYME_DUP, local_f2.data(), df2_val.data(),
            TRIBOL_ENZYME_CONST, size2
        );

        for (int j = 0; j < size1 * Dim; ++j) df2_dp(j, i) = df1_val[j];
        for (int j = 0; j < size2 * Dim; ++j) df1_dp(j, i) = df2_val[j];
        p1_dot[i] = 0.0;
    }

    // Primal values
    ForceModel::compute(x1, n1, p1, local_f1.data(), g1.data(), size1, x2, local_f2.data(), size2);
    
    f1.SetSize(size2 * Dim);
    f2.SetSize(size1 * Dim);
    
    for(int i=0; i<size2*Dim; ++i) f1(i) = local_f2[i];
    for(int i=0; i<size1*Dim; ++i) f2(i) = local_f1[i];

    energy = 0.0; // Not computed
  }

  void computeForce( const InterfacePair& pair, const IntegrationRule<ForceModel, Dim>& integration_rule,
                     ForceModel& gap_method, const std::vector<RealT>& coords1, const std::vector<RealT>& coords2,
                     std::shared_ptr<ArrayViewT<RealT>> pressures, mfem::Vector& f1, mfem::Vector& f2,
                     RealT& energy ) override
  {
    const int size1 = coords2.size() / Dim; // Nonmortar
    const int size2 = coords1.size() / Dim; // Mortar
    const RealT* x1 = coords2.data();
    const RealT* x2 = coords1.data();
    const RealT* n1 = gap_method.n1.data();
    const RealT* p1 = (pressures && pressures->size() > 0) ? pressures->data() : nullptr;
    std::vector<RealT> p1_dummy(size1, 0.0);
    if (!p1) p1 = p1_dummy.data();

    std::vector<RealT> local_f1(size1 * Dim);
    std::vector<RealT> local_f2(size2 * Dim);
    std::vector<RealT> g1(size1);

    ForceModel::compute(x1, n1, p1, local_f1.data(), g1.data(), size1, x2, local_f2.data(), size2);

    f1.SetSize(size2 * Dim);
    f2.SetSize(size1 * Dim);
    for(int i=0; i<size2*Dim; ++i) f1(i) = local_f2[i];
    for(int i=0; i<size1*Dim; ++i) f2(i) = local_f1[i];
    
    energy = 0.0;
  }
};
#endif

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ELEMENTFORCEANDGAPENZYME_HPP_ */
