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
   
   for (int i = 0; i < pmesh.GetNBE(); ++i) {
       FaceElementTransformations* tr = pmesh.GetBdrFaceTransformations(i);
       if (tr) {
           int mask = tr->GetConfigurationMask();
           int elem_id = (mask & FaceElementTransformations::HAVE_ELEM1) ? tr->Elem1No : tr->Elem2No;
           std::cout << "Rank " << myid << ": Boundary element " << i << " -> Volume element " << elem_id << std::endl;
       }
   }

   return 0;
}
