#include "mfem.hpp"
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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
      std::cout << "Rank " << myid << ": Dof_TrueDof_Matrix exists. Size: " << P_mat->Height() << "x" << P_mat->Width() << std::endl;
   } else {
      std::cout << "Rank " << myid << ": Dof_TrueDof_Matrix is NULL." << std::endl;
   }

   const Operator *P = pfes.GetProlongationMatrix();
   if (P) {
      std::cout << "Rank " << myid << ": Prolongation exists. Size: " << P->Height() << "x" << P->Width() << std::endl;
   } else {
      std::cout << "Rank " << myid << ": Prolongation is NULL." << std::endl;
   }

   MPI_Finalize();
   return 0;
}
