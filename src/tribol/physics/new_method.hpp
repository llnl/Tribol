#pragma once
#include <vector>
#include <array>

struct Node {
    double x, y;
    int id;
};

struct Element {
    int id;
    std::array<int, 2> node_ids;
};

struct Mesh {
    std::vector<Node> nodes; 
    std::vector<Element> elements; 

    const Node& node(int i) const { return nodes[i]; }
    Node& node(int i) {return nodes[i]; }
};

struct QuadPoints {
    std::vector<double> qp; //quadpoints 
    std::vector<double> w; //weights
};

struct ContactParams {
    double del;
    double k;
    int N;
};

struct NodalContactData {
    std::array<double, 2> AI;
    std::array<double, 2> g_tilde;
};

struct FDResult {
    std::array<double, 2> dgt;
};

struct FiniteDiffResult {
    std::vector<double> fd_gradient_g1;
    std::vector<double> fd_gradient_g2;
    std::vector<double> analytical_gradient_g1;
    std::vector<double> analytical_gradient_g2;
    std::vector<int> node_ids;
    double g_tilde1_baseline;
    double g_tilde2_baseline;
};

class ContactSmoothing {
    public:
        explicit ContactSmoothing (const ContactParams& p) : p_(p) {} //Constructor 

        std::array<double, 2> bounds_from_projections(const std::array<double, 2>& proj) const;

        std::array<double, 2> smooth_bounds(const std::array<double, 2>& bounds) const;

    private: 
        ContactParams p_;

};

class ContactEvaluator {
    public: 
        explicit ContactEvaluator(const ContactParams& p) 
        : p_(p), smoother_(p) {} //constructor - copies params into the object 



        double compute_contact_energy(const Mesh& mesh, 
                      const Element& A, 
                      const Element& B) const;

        void grad_gtilde(const Mesh& mesh, const Element& A, const Element& B,
                         double dgt1_dx[8], double dgt2_dx[8]) const;

        void grad_trib_area(const Mesh& mesh, const Element& A, const Element& B, 
                            double dA1_dx[8], double dA2_dx[8]) const; 

        void d2_g2tilde(const Mesh& mesh, const Element& A, const Element& B,
                        double dgt1_dx[64], double dgt2_dx[64]) const;

        void compute_d2A_d2u(const Mesh& mesh, const Element& A, const Element& B,
                             double dgt1_dx[64], double dgt2_dx[64]) const;

        std::array<double, 8> compute_contact_forces(const Mesh& mesh, const Element& A, const Element& B) const;

        std::array<std::array<double, 8>, 8> compute_stiffness_matrix(const Mesh& mesh, const Element& A, const Element& B) const;

        std::pair<double, double> eval_gtilde(const Mesh& mesh,
                                              const Element& A, 
                                              const Element& B) const;

        FiniteDiffResult validate_g_tilde(Mesh& mesh,
                                          const Element& A,
                                          const Element& B,
                                          double epsilon = 1e-7) const;
        
        void print_gradient_comparison(const FiniteDiffResult& val) const;
        
        std::pair<double,double> eval_gtilde_fixed_qp(Mesh& mesh,
                                                      const Element& A,
                                                      const Element& B,
                                                      const QuadPoints& qp_fixed) const;

        FiniteDiffResult validate_hessian(Mesh& mesh, 
                                          const Element& A, 
                                          const Element& B, 
                                          double epsilon = 1e-7) const;

        void grad_gtilde_with_qp(const Mesh& mesh, const Element& A, const Element& B,
                            const QuadPoints& qp_fixed, 
                            double dgt1_dx[8], double dgt2_dx[8]) const;

        void print_hessian_comparison(const FiniteDiffResult& val) const;

    private: 
        ContactParams p_;
        ContactSmoothing smoother_;
        QuadPoints compute_quadrature(const std::array<double,2>& xi_bounds) const;

        std::array<double, 2> projections(const Mesh& mesh,
                                          const Element& A, 
                                          const Element& B) const; 

        double gap(const Mesh& mesh,
                   const Element& A, 
                   const Element& B,
                   double xiA) const;

        NodalContactData compute_nodal_contact_data(const Mesh& mesh,
                                                         const Element& A, 
                                                         const Element& B) const; 
        
        std::array<double, 2> compute_pressures(const NodalContactData& ncd) const;
        

                                       
};
