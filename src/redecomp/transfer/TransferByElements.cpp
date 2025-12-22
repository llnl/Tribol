// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferByElements.hpp"

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"
#include "shared/LoopExec.hpp"

#include "redecomp/RedecompMesh.hpp"

#if defined(TRIBOL_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace redecomp {

using namespace tribol;

void TransferByElements::TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // checks to make sure src and dst are valid
  auto redecomp = dynamic_cast<RedecompMesh*>( dst.FESpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The Mesh of GridFunction dst must be a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( src.ParFESpace()->GetParMesh() != &redecomp->getParent(),
                      "The Meshes of the specified GridFunctions are not related in a "
                      "Redecomp -> ParMesh relationship." );
  SLIC_ERROR_ROOT_IF( strcmp( src.FESpace()->FEColl()->Name(), dst.FESpace()->FEColl()->Name() ) != 0,
                      "The FiniteElementCollections of the specified GridFunctions are not "
                      "the same." );
  SLIC_ERROR_ROOT_IF( src.FESpace()->GetVDim() != dst.FESpace()->GetVDim(),
                      "The vdim of the FiniteElementSpaces of the specified GridFunctions are "
                      "not the same." );

  bool use_gpu = false;
#if defined(TRIBOL_USE_CUDA) || defined(TRIBOL_USE_HIP)
  use_gpu = (src.GetMemory().GetMemoryType() != mfem::MemoryType::HOST);
#endif

  auto exec_mode = use_gpu ? ExecutionMode::Cuda : ExecutionMode::Sequential;
  int allocator_id = use_gpu ? getResourceAllocatorID( MemorySpace::Device ) : getDefaultAllocatorID();
  int host_allocator_id = getDefaultAllocatorID();

  int vdim = src.FESpace()->GetVDim();
  bool src_by_nodes = ( src.FESpace()->GetOrdering() == mfem::Ordering::byNODES );

  // Setup connectivity on device if needed
  axom::Array<int> el_dof_ip;
  axom::Array<int> el_dof_j;
  const double* src_data = nullptr;

  if ( use_gpu ) {
     const auto& el_dof = src.FESpace()->GetElementToDofTable();
     el_dof_ip = axom::Array<int>( el_dof.Size() + 1, el_dof.Size() + 1, allocator_id );
     axom::copy( el_dof_ip.data(), el_dof.GetI(), (el_dof.Size() + 1) * sizeof(int) );
     el_dof_j = axom::Array<int>( el_dof.Size_of_connections(), el_dof.Size_of_connections(), allocator_id );
     axom::copy( el_dof_j.data(), el_dof.GetJ(), el_dof.Size_of_connections() * sizeof(int) );
     src_data = src.Read();
#if defined(TRIBOL_USE_CUDA)
     cudaDeviceSynchronize();
#endif
  }

  // send and receive DOF values from other ranks
  auto dst_dofs = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach( [redecomp, &src, use_gpu, exec_mode, allocator_id, host_allocator_id, &el_dof_ip, &el_dof_j, src_data, vdim, src_by_nodes]( int dest ) {
    auto src_dofs = axom::Array<double>();
    const auto& src_elem_idx = redecomp->getParentToRedecompElems().first[dest];
    auto n_els = src_elem_idx.size();
    
    if ( n_els > 0 ) {
      if ( use_gpu ) {
        // GPU Packing
        axom::Array<int> dev_elem_idx( src_elem_idx, allocator_id );
        
        // Calculate total size and offsets on Host
        const int* I = src.FESpace()->GetElementToDofTable().GetI();
        axom::Array<int> host_offsets( n_els + 1 );
        int current_offset = 0;
        host_offsets[0] = 0;
        for ( int k = 0; k < (int)n_els; ++k ) {
           int e = src_elem_idx[k];
           current_offset += (I[e+1] - I[e]) * vdim;
           host_offsets[k+1] = current_offset;
        }

        axom::Array<double> dev_src_dofs( current_offset, current_offset, allocator_id );
        auto* src_dofs_ptr = dev_src_dofs.data();
        auto* elem_idx_ptr = dev_elem_idx.data();
        auto* I_ptr = el_dof_ip.data();
        auto* J_ptr = el_dof_j.data();

        axom::Array<int> dev_offsets( host_offsets, allocator_id );
        auto* offsets_ptr = dev_offsets.data();

        int ndofs = src.FESpace()->GetNDofs();

        forAllExec( exec_mode, n_els, [=] TRIBOL_DEVICE ( int k ) {
           int e = elem_idx_ptr[k];
           int start_dof = I_ptr[e];
           int end_dof = I_ptr[e+1];
           int count = end_dof - start_dof;
           int write_offset = offsets_ptr[k];
           
           for ( int i = 0; i < count; ++i ) {
              int dof_idx = J_ptr[ start_dof + i ];
              for ( int v = 0; v < vdim; ++v ) {
                 int vdof_idx = src_by_nodes ? (dof_idx + v * ndofs) : (dof_idx * vdim + v);
                 src_dofs_ptr[ write_offset + i*vdim + v ] = src_data[ vdof_idx ];
              }
           }
        });
#if defined(TRIBOL_USE_CUDA)
        cudaDeviceSynchronize();
#endif
        // Copy to Host for MPI
        src_dofs = axom::Array<double>( current_offset, current_offset, host_allocator_id );
        axom::copy( src_dofs.data(), dev_src_dofs.data(), current_offset * sizeof(double) );
      } else {
        // Host Packing (Original)
        auto elem_vdofs = mfem::Array<int>();
        auto dof_vals = mfem::Vector();
        src.FESpace()->GetElementVDofs( src_elem_idx[0], elem_vdofs );
        src_dofs.reserve( elem_vdofs.Size() * n_els );
        auto vdof_ct = 0;
        for ( int e{ 0 }; e < (int)n_els; ++e ) {
          src.FESpace()->GetElementVDofs( src_elem_idx[e], elem_vdofs );
          src.GetSubVector( elem_vdofs, dof_vals );
          src_dofs.insert( vdof_ct, dof_vals.Size(), dof_vals.GetData() );
          vdof_ct += dof_vals.Size();
        }
      }
    }
    return src_dofs;
  } );

  // map received DOF values to local DOFs
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  
  // Note: Check DST memory type for unpacking
  bool dst_use_gpu = false;
#if defined(TRIBOL_USE_CUDA) || defined(TRIBOL_USE_HIP)
  dst_use_gpu = (dst.GetMemory().GetMemoryType() != mfem::MemoryType::HOST);
#endif
  
  if ( dst_use_gpu ) {
     // Setup dst connectivity
     const auto& dst_el_dof = dst.FESpace()->GetElementToDofTable();
     axom::Array<int> dst_el_dof_ip( dst_el_dof.Size() + 1, dst_el_dof.Size() + 1, allocator_id );
     axom::copy( dst_el_dof_ip.data(), dst_el_dof.GetI(), (dst_el_dof.Size() + 1) * sizeof(int) );
     axom::Array<int> dst_el_dof_j( dst_el_dof.Size_of_connections(), dst_el_dof.Size_of_connections(), allocator_id );
     axom::copy( dst_el_dof_j.data(), dst_el_dof.GetJ(), dst_el_dof.Size_of_connections() * sizeof(int) );
     double* dst_data = dst.Write();
#if defined(TRIBOL_USE_CUDA)
     cudaDeviceSynchronize();
#endif
     
     auto* dst_I_ptr = dst_el_dof_ip.data();
     auto* dst_J_ptr = dst_el_dof_j.data();

     int ndofs = dst.FESpace()->GetNDofs();
     bool dst_by_nodes = ( dst.FESpace()->GetOrdering() == mfem::Ordering::byNODES );

     for ( int r{ 0 }; r < n_ranks; ++r ) {
        auto first_el = redecomp->getRedecompToParentElemOffsets()[r];
        auto last_el = redecomp->getRedecompToParentElemOffsets()[r + 1];
        int n_els = last_el - first_el;
        if (n_els == 0) continue;
        
        // Data to unpack
        const auto& recv_data = dst_dofs[r];
        if ( recv_data.size() == 0 ) continue;
        
        axom::Array<double> dev_recv_data( recv_data.size(), recv_data.size(), allocator_id );
        axom::copy( dev_recv_data.data(), recv_data.data(), recv_data.size() * sizeof(double) );
        auto* recv_ptr = dev_recv_data.data();
        
        // Offsets
        const int* dst_I_host = dst.FESpace()->GetElementToDofTable().GetI();
        axom::Array<int> host_offsets( n_els + 1 );
        int current_offset = 0;
        host_offsets[0] = 0;
        for ( int e = first_el; e < last_el; ++e ) {
            current_offset += (dst_I_host[e+1] - dst_I_host[e]) * vdim;
            host_offsets[e - first_el + 1] = current_offset;
        }
        axom::Array<int> dev_offsets( host_offsets, allocator_id );
        auto* offsets_ptr = dev_offsets.data();

        forAllExec( exec_mode, n_els, [=] TRIBOL_DEVICE ( int k ) {
           int e = first_el + k; 
           int start_dof = dst_I_ptr[e];
           int end_dof = dst_I_ptr[e+1];
           int count = end_dof - start_dof;
           int read_offset = offsets_ptr[k];
           
           for ( int i = 0; i < count; ++i ) {
              int dof_idx = dst_J_ptr[ start_dof + i ];
              for ( int v = 0; v < vdim; ++v ) {
                 int vdof_idx = dst_by_nodes ? (dof_idx + v * ndofs) : (dof_idx * vdim + v);
                 dst_data[ vdof_idx ] = recv_ptr[ read_offset + i*vdim + v ];
              }
           }
        });
     }
#if defined(TRIBOL_USE_CUDA)
     cudaDeviceSynchronize();
#endif
  } else {
      // Host Unpacking
      auto elem_vdofs = mfem::Array<int>();
      for ( int r{ 0 }; r < n_ranks; ++r ) {
        auto vdof_ct = 0;
        auto first_el = redecomp->getRedecompToParentElemOffsets()[r];
        auto last_el = redecomp->getRedecompToParentElemOffsets()[r + 1];
        for ( int e{ first_el }; e < last_el; ++e ) {
          dst.FESpace()->GetElementVDofs( e, elem_vdofs );
          auto dof_vals = mfem::Vector( &dst_dofs[r][vdof_ct], elem_vdofs.Size() );
          dst.SetSubVector( elem_vdofs, dof_vals );
          vdof_ct += elem_vdofs.Size();
        }
      }
  }
}

void TransferByElements::TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // checks to make sure src and dst are valid
  auto redecomp = dynamic_cast<RedecompMesh*>( src.FESpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The Mesh of GridFunction dst must be a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( dst.ParFESpace()->GetParMesh() != &redecomp->getParent(),
                      "The Meshes of the specified GridFunctions are not related in a"
                      "Redecomp -> ParMesh relationship." );
  SLIC_ERROR_ROOT_IF( strcmp( dst.FESpace()->FEColl()->Name(), src.FESpace()->FEColl()->Name() ) != 0,
                      "The FiniteElementCollections of the specified GridFunctions are not"
                      "the same." );
  SLIC_ERROR_ROOT_IF( dst.FESpace()->GetVDim() != src.FESpace()->GetVDim(),
                      "The vdim of the FiniteElementSpaces of the specified GridFunctions are"
                      "not the same." );

  bool use_gpu = false;
#if defined(TRIBOL_USE_CUDA) || defined(TRIBOL_USE_HIP)
  use_gpu = (src.GetMemory().GetMemoryType() != mfem::MemoryType::HOST);
#endif

  auto exec_mode = use_gpu ? ExecutionMode::Cuda : ExecutionMode::Sequential;
  int allocator_id = use_gpu ? getResourceAllocatorID( MemorySpace::Device ) : getDefaultAllocatorID();
  int host_allocator_id = getDefaultAllocatorID();

  int vdim = src.FESpace()->GetVDim();
  bool src_by_nodes = ( src.FESpace()->GetOrdering() == mfem::Ordering::byNODES );

  // Setup connectivity on device if needed
  axom::Array<int> el_dof_ip;
  axom::Array<int> el_dof_j;
  const double* src_data = nullptr;

  if ( use_gpu ) {
     const auto& el_dof = src.FESpace()->GetElementToDofTable();
     el_dof_ip = axom::Array<int>( el_dof.Size() + 1, el_dof.Size() + 1, allocator_id );
     axom::copy( el_dof_ip.data(), el_dof.GetI(), (el_dof.Size() + 1) * sizeof(int) );
     el_dof_j = axom::Array<int>( el_dof.Size_of_connections(), el_dof.Size_of_connections(), allocator_id );
     axom::copy( el_dof_j.data(), el_dof.GetJ(), el_dof.Size_of_connections() * sizeof(int) );
     src_data = src.Read();
#if defined(TRIBOL_USE_CUDA)
     cudaDeviceSynchronize();
#endif
  }

  // send and receive non-ghost DOF values from other ranks
  auto dst_dofs = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach( [redecomp, &src, use_gpu, exec_mode, allocator_id, host_allocator_id, &el_dof_ip, &el_dof_j, src_data, vdim, src_by_nodes]( int dest ) {
    auto src_dofs = axom::Array<double>();
    auto first_el = redecomp->getRedecompToParentElemOffsets()[dest];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[dest + 1];
    auto n_els = last_el - first_el;
    
    if ( n_els > 0 ) {
      if ( use_gpu ) {
         const auto& ghosts = redecomp->getRedecompToParentGhostElems()[dest];
         axom::Array<int> valid_elems; 
         valid_elems.reserve( n_els );
         int ghost_ct = 0;
         for ( int e{ first_el }; e < last_el; ++e ) {
             if ( ghost_ct < (int)ghosts.size() && ghosts[ghost_ct] == e ) {
                 ++ghost_ct;
             } else {
                 valid_elems.push_back(e);
             }
         }
         
         if ( valid_elems.empty() ) return src_dofs;

         int n_valid = (int)valid_elems.size();
         const int* I = src.FESpace()->GetElementToDofTable().GetI();
         axom::Array<int> host_offsets( n_valid + 1 );
         int total_size = 0;
         host_offsets[0] = 0;
         for ( int k = 0; k < n_valid; ++k ) {
             int e = valid_elems[k];
             total_size += (I[e+1] - I[e]) * vdim;
             host_offsets[k+1] = total_size;
         }
         
         axom::Array<double> dev_src_dofs( total_size, total_size, allocator_id );
         axom::Array<int> dev_elem_idx( valid_elems, allocator_id );
         axom::Array<int> dev_offsets( host_offsets, allocator_id );
         
         auto* src_dofs_ptr = dev_src_dofs.data();
         auto* elem_idx_ptr = dev_elem_idx.data();
         auto* I_ptr = el_dof_ip.data();
         auto* J_ptr = el_dof_j.data();
         auto* offsets_ptr = dev_offsets.data();

         int ndofs = src.FESpace()->GetNDofs();
         
         forAllExec( exec_mode, n_valid, [=] TRIBOL_DEVICE ( int k ) {
            int e = elem_idx_ptr[k];
            int start_dof = I_ptr[e];
            int end_dof = I_ptr[e+1];
            int count = end_dof - start_dof;
            int write_offset = offsets_ptr[k];
            
            for ( int i = 0; i < count; ++i ) {
               int dof_idx = J_ptr[ start_dof + i ];
               for ( int v = 0; v < vdim; ++v ) {
                  int vdof_idx = src_by_nodes ? (dof_idx + v * ndofs) : (dof_idx * vdim + v);
                  src_dofs_ptr[ write_offset + i*vdim + v ] = src_data[ vdof_idx ];
               }
            }
         });
#if defined(TRIBOL_USE_CUDA)
         cudaDeviceSynchronize();
#endif
         // Copy to Host for MPI
         src_dofs = axom::Array<double>( total_size, total_size, host_allocator_id );
         axom::copy( src_dofs.data(), dev_src_dofs.data(), total_size * sizeof(double) );
      } else {
          // Host Packing
          auto elem_vdofs = mfem::Array<int>();
          auto dof_vals = mfem::Vector();
          src.FESpace()->GetElementVDofs( first_el, elem_vdofs );
          src_dofs.reserve( elem_vdofs.Size() * n_els );
          auto vdof_ct = 0;
          auto ghost_ct = 0;
          for ( int e{ first_el }; e < last_el; ++e ) {
            if ( ghost_ct < (int)redecomp->getRedecompToParentGhostElems()[dest].size() &&
                 redecomp->getRedecompToParentGhostElems()[dest][ghost_ct] == e ) {
              ++ghost_ct;
            } else {
              src.FESpace()->GetElementVDofs( e, elem_vdofs );
              src.GetSubVector( elem_vdofs, dof_vals );
              src_dofs.insert( vdof_ct, dof_vals.Size(), dof_vals.GetData() );
              vdof_ct += dof_vals.Size();
            }
          }
      }
    }
    return src_dofs;
  } );

  // map received non-ghost DOF values to local DOFs
  auto n_ranks = redecomp->getMPIUtility().NRanks();

  // Note: Check DST memory type for unpacking
  bool dst_use_gpu = false;
#if defined(TRIBOL_USE_CUDA) || defined(TRIBOL_USE_HIP)
  dst_use_gpu = (dst.GetMemory().GetMemoryType() != mfem::MemoryType::HOST);
#endif
  
  if ( dst_use_gpu ) {
      const auto& dst_el_dof = dst.FESpace()->GetElementToDofTable();
      axom::Array<int> dst_el_dof_ip( dst_el_dof.Size() + 1, dst_el_dof.Size() + 1, allocator_id );
      axom::copy( dst_el_dof_ip.data(), dst_el_dof.GetI(), (dst_el_dof.Size() + 1) * sizeof(int) );
      axom::Array<int> dst_el_dof_j( dst_el_dof.Size_of_connections(), dst_el_dof.Size_of_connections(), allocator_id );
      axom::copy( dst_el_dof_j.data(), dst_el_dof.GetJ(), dst_el_dof.Size_of_connections() * sizeof(int) );
      double* dst_data = dst.Write();
#if defined(TRIBOL_USE_CUDA)
      cudaDeviceSynchronize();
#endif
      
      auto* dst_I_ptr = dst_el_dof_ip.data();
      auto* dst_J_ptr = dst_el_dof_j.data();

      int ndofs = dst.FESpace()->GetNDofs();
      bool dst_by_nodes = ( dst.FESpace()->GetOrdering() == mfem::Ordering::byNODES );
      
      for ( int r{ 0 }; r < n_ranks; ++r ) {
         const auto& elem_list = redecomp->getParentToRedecompElems().first[r];
         const auto& is_ghost = redecomp->getParentToRedecompElems().second[r];
         
         axom::Array<int> valid_elems;
         valid_elems.reserve( elem_list.size() );
         for( int k=0; k < (int)elem_list.size(); ++k ) {
             if ( !is_ghost[k] ) {
                 valid_elems.push_back( elem_list[k] );
             }
         }
         
         if ( valid_elems.empty() ) continue;
         
         int n_valid = (int)valid_elems.size();
         const auto& recv_data = dst_dofs[r];
         if ( recv_data.size() == 0 ) continue;
         
         // Copy Host -> Device
         axom::Array<double> dev_recv_data( recv_data.size(), recv_data.size(), allocator_id );
         axom::copy( dev_recv_data.data(), recv_data.data(), recv_data.size() * sizeof(double) );
         auto* recv_ptr = dev_recv_data.data();
         
         const int* dst_I_host = dst.FESpace()->GetElementToDofTable().GetI();
         axom::Array<int> host_offsets( n_valid + 1 );
         int total_size = 0;
         host_offsets[0] = 0;
         for ( int k = 0; k < n_valid; ++k ) {
             int e = valid_elems[k];
             total_size += (dst_I_host[e+1] - dst_I_host[e]) * vdim;
             host_offsets[k+1] = total_size;
         }
         axom::Array<int> dev_elem_idx( valid_elems, allocator_id );
         axom::Array<int> dev_offsets( host_offsets, allocator_id );
         auto* offsets_ptr = dev_offsets.data();
         auto* elem_idx_ptr = dev_elem_idx.data();
         
         forAllExec( exec_mode, n_valid, [=] TRIBOL_DEVICE ( int k ) {
             int e = elem_idx_ptr[k];
             int start_dof = dst_I_ptr[e];
             int end_dof = dst_I_ptr[e+1];
             int count = end_dof - start_dof;
             int read_offset = offsets_ptr[k];
             
             for ( int i = 0; i < count; ++i ) {
                 int dof_idx = dst_J_ptr[ start_dof + i ];
                 for ( int v = 0; v < vdim; ++v ) {
                    int vdof_idx = dst_by_nodes ? (dof_idx + v * ndofs) : (dof_idx * vdim + v);
                    dst_data[ vdof_idx ] = recv_ptr[ read_offset + i*vdim + v ];
                 }
             }
         });
      }
#if defined(TRIBOL_USE_CUDA)
      cudaDeviceSynchronize();
#endif

  } else {
      // Host Unpacking
      auto elem_vdofs = mfem::Array<int>();
      for ( int r{ 0 }; r < n_ranks; ++r ) {
        auto vdof_ct = 0;
        for ( int e{ 0 }; e < (int)redecomp->getParentToRedecompElems().first[r].size(); ++e ) {
          if ( !redecomp->getParentToRedecompElems().second[r][e] ) {
            dst.FESpace()->GetElementVDofs( redecomp->getParentToRedecompElems().first[r][e], elem_vdofs );
            auto dof_vals = mfem::Vector( &dst_dofs[r][vdof_ct], elem_vdofs.Size() );
            dst.SetSubVector( elem_vdofs, dof_vals );
            vdof_ct += elem_vdofs.Size();
          }
        }
      }
  }
}

}  // end namespace redecomp