#include "mfem.hpp"
#include <iostream>
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL);
    mesh.EnsureNCMesh(true);
    if (mfem::Mpi::WorldRank() == 0) {
        mfem::Array<int> ref(1); ref[0] = 0;
        mesh.GeneralRefinement(ref);
    }
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::H1_FECollection fec(1, 2);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec);
    
    auto P = fes.GetProlongationMatrix();
    auto P_mat = dynamic_cast<const mfem::HypreParMatrix*>(P);
    
    if (P_mat) {
        std::cout << "P is HypreParMatrix" << std::endl;
    } else {
        std::cout << "P is not HypreParMatrix" << std::endl;
    }
    MPI_Finalize();
    return 0;
}
