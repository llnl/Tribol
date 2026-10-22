#include "mfem.hpp"
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   mesh.EnsureNCMesh(true);
   
   if (myid == 0) {
      Array<int> ref(1); ref[0] = 0;
      mesh.GeneralRefinement(ref);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   
   Array<HYPRE_BigInt> gv;
   pmesh.GetGlobalVertexIndices(gv);
   
   for (int i = 0; i < pmesh.GetNV(); ++i) {
       std::cout << "Rank " << myid << ": vertex " << i << " global vertex ID: " << gv[i] << std::endl;
   }

   MPI_Finalize();
   return 0;
}