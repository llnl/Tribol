#include "tribol/execution/CudaPenalty.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tribol::execution {

namespace {

void requireCuda( cudaError_t error, const char* operation )
{
  if ( error != cudaSuccess ) {
    throw std::runtime_error( std::string( operation ) + ": " + cudaGetErrorString( error ) );
  }
}

__global__ void penaltyKernel( ArrayView<const InteractionPatch> patches, Real stiffness,
                               ArrayView<PenaltyContribution> contributions )
{
  const Index patch = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( patch < patches.size() ) {
    contributions[patch] = evaluatePenaltyPatch( patches[patch], stiffness, integration::Polygon<2>{} );
  }
}

__global__ void pointwisePenaltyKernel( ArrayView<const InteractionPatch> patches, Real stiffness,
                                        constraint::GapActivationParameters activation,
                                        ArrayView<PenaltyContribution> contributions )
{
  const Index patch = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( patch < patches.size() ) {
    contributions[patch] = evaluatePointwisePenaltyPatch( patches[patch], stiffness, activation );
  }
}

void requireValidArguments( ArrayView<const InteractionPatch> patches, Real stiffness,
                            ArrayView<PenaltyContribution> contributions )
{
  if ( stiffness < 0.0 || contributions.size() != patches.size() || !patches.isStructurallyValid() ||
       !contributions.isStructurallyValid() ) {
    throw std::invalid_argument( "CUDA penalty execution requires nonnegative stiffness and matching valid views." );
  }
}

}  // namespace

bool cudaDeviceAvailable()
{
  int count{};
  return cudaGetDeviceCount( &count ) == cudaSuccess && count > 0;
}

CudaPenaltyWorkspace::~CudaPenaltyWorkspace()
{
  cudaFree( device_contributions_ );
  cudaFree( device_patches_ );
}

void CudaPenaltyWorkspace::reserve( Index patch_capacity )
{
  if ( patch_capacity <= capacity_ ) {
    return;
  }
  InteractionPatch* new_patches{};
  PenaltyContribution* new_contributions{};
  const std::size_t patch_bytes = static_cast<std::size_t>( patch_capacity ) * sizeof( InteractionPatch );
  const std::size_t contribution_bytes = static_cast<std::size_t>( patch_capacity ) * sizeof( PenaltyContribution );
  requireCuda( cudaMalloc( &new_patches, patch_bytes ), "cudaMalloc interaction patches" );
  try {
    requireCuda( cudaMalloc( &new_contributions, contribution_bytes ), "cudaMalloc penalty contributions" );
  } catch ( ... ) {
    cudaFree( new_patches );
    throw;
  }
  cudaFree( device_contributions_ );
  cudaFree( device_patches_ );
  device_patches_ = new_patches;
  device_contributions_ = new_contributions;
  capacity_ = patch_capacity;
}

void CudaPenaltyWorkspace::evaluatePointwise( ArrayView<const InteractionPatch> patches, Real stiffness,
                                              constraint::GapActivationParameters activation,
                                              ArrayView<PenaltyContribution> contributions )
{
  requireValidArguments( patches, stiffness, contributions );
  if ( patches.empty() ) {
    return;
  }
  reserve( patches.size() );
  const std::size_t patch_bytes = static_cast<std::size_t>( patches.size() ) * sizeof( InteractionPatch );
  const std::size_t contribution_bytes =
      static_cast<std::size_t>( contributions.size() ) * sizeof( PenaltyContribution );
  requireCuda( cudaMemcpy( device_patches_, patches.data(), patch_bytes, cudaMemcpyHostToDevice ),
               "cudaMemcpy interaction patches" );
  constexpr int block_size = 128;
  const int blocks = static_cast<int>( ( patches.size() + block_size - 1 ) / block_size );
  pointwisePenaltyKernel<<<blocks, block_size>>>( { device_patches_, patches.size() }, stiffness, activation,
                                                  { device_contributions_, contributions.size() } );
  requireCuda( cudaGetLastError(), "launch pointwise penalty kernel" );
  requireCuda( cudaDeviceSynchronize(), "synchronize pointwise penalty kernel" );
  requireCuda( cudaMemcpy( contributions.data(), device_contributions_, contribution_bytes, cudaMemcpyDeviceToHost ),
               "cudaMemcpy pointwise penalty contributions" );
}

void evaluatePointwisePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                                      ArrayView<PenaltyContribution> contributions, Cuda,
                                      constraint::GapActivationParameters activation )
{
  CudaPenaltyWorkspace workspace;
  workspace.evaluatePointwise( patches, stiffness, activation, contributions );
}

void evaluatePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                             ArrayView<PenaltyContribution> contributions, Cuda, integration::Polygon<2> )
{
  requireValidArguments( patches, stiffness, contributions );
  if ( !cudaDeviceAvailable() ) {
    throw std::runtime_error( "CUDA execution was requested, but no CUDA device is available." );
  }
  InteractionPatch* device_patches{};
  PenaltyContribution* device_contributions{};
  const std::size_t patch_bytes = static_cast<std::size_t>( patches.size() ) * sizeof( InteractionPatch );
  const std::size_t contribution_bytes =
      static_cast<std::size_t>( contributions.size() ) * sizeof( PenaltyContribution );
  requireCuda( cudaMalloc( &device_patches, patch_bytes ), "cudaMalloc interaction patches" );
  try {
    requireCuda( cudaMalloc( &device_contributions, contribution_bytes ), "cudaMalloc penalty contributions" );
    requireCuda( cudaMemcpy( device_patches, patches.data(), patch_bytes, cudaMemcpyHostToDevice ),
                 "cudaMemcpy interaction patches" );
    constexpr int block_size = 128;
    const int blocks = static_cast<int>( ( patches.size() + block_size - 1 ) / block_size );
    penaltyKernel<<<blocks, block_size>>>( { device_patches, patches.size() }, stiffness,
                                           { device_contributions, contributions.size() } );
    requireCuda( cudaGetLastError(), "launch penalty kernel" );
    requireCuda( cudaDeviceSynchronize(), "synchronize penalty kernel" );
    requireCuda( cudaMemcpy( contributions.data(), device_contributions, contribution_bytes, cudaMemcpyDeviceToHost ),
                 "cudaMemcpy penalty contributions" );
  } catch ( ... ) {
    cudaFree( device_contributions );
    cudaFree( device_patches );
    throw;
  }
  requireCuda( cudaFree( device_contributions ), "cudaFree penalty contributions" );
  requireCuda( cudaFree( device_patches ), "cudaFree interaction patches" );
}

}  // namespace tribol::execution
