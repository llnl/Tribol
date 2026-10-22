#include "mfem.hpp"
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   mesh.EnsureNCMesh(true);
   
   if (myid == 0) {
      Array<int> ref(1); ref[0] = 0;
      mesh.GeneralRefinement(ref);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(1, 2);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   HypreParMatrix *P_mat = pfes.Dof_TrueDof_Matrix();
   if (P_mat) {
      std::cout << "Rank " << myid << ": Dof_TrueDof_Matrix exists." << std::endl;
      const Operator *P_op = pfes.GetProlongationMatrix();
      if (P_op != P_mat) {
          std::cout << "Rank " << myid << ": ProlongationMatrix is different from Dof_TrueDof_Matrix." << std::endl;
      }
   } else {
      std::cout << "Rank " << myid << ": Dof_TrueDof_Matrix is NULL." << std::endl;
   }

   return 0;
}
