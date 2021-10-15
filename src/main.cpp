#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>

#include "matrix.hpp"
#include "gcn.hpp"

#include <chrono>
#include <thread>
#include <vector>
#include <string>

using namespace std;

/** Helper class to wrap and capture argument errors */
class arg_error : public ::std::runtime_error {
    public:
        template<typename T>
            arg_error(T t) : ::std::runtime_error(t) { }
};

/** Helper function for printing the program usage line to cout */
int usage_(char* prog) {
    cout << "Usage: " << prog << " [-h] [-v] [-P num] [command, ...]" << endl;
    return 0;
}

/** Helper function for printing out a help message */
int help_() {
    cout << "\n"
        "MG-GCN is a multi-GPU GCN training framework\n"
        "This software implements MG-GCN using CUDA.\n\n"
        "Options:\n"
        "    -P : a number to print\n\n"
        "    -R : Enable row partition\n\n"
        "Arguments:\n"
        "    command : the command\n"
        ;
    return EXIT_SUCCESS;
}

using x_t = unsigned;
using v_t = unsigned;
using r_t = float;

int main_(int argc, char** argv) {
    opterr = 0;

    std::size_t P = 1;
    std::size_t row_partition = 0;
    std::size_t num_epochs = 20;
    bool overlap = true;
    bool no_wait = false;
    // Parse the command line arguments
    while (optind < argc) {
        char c = '\0';
        if ((c = getopt(argc, argv, "h?P:R:E:S:N")) != -1) {
            // Handle the case arguments
            switch (c) {
                case '?': { return usage_(argv[0]); }
                case 'h': { usage_(argv[0]); return help_(); }
                case 'P': { P = std::stoull(optarg); break; }
                case 'R': { row_partition = std::stoull(optarg); break; }
                case 'E': { num_epochs = std::stoull(optarg); break; }
                case 'S': { overlap = false; break; }
                case 'N': { no_wait = true; overlap = true; break; }
                default: { throw arg_error("Unknown argument."); }
            }
        } else
            break;
    }

    // Handle the program commands
    while (argv[optind] != NULL) {
        // Register the command
        string command = argv[optind++];

        if (std::string(command.begin(), command.begin() + 5) == "train") {
            std::filesystem::path dir = argv[optind++];

            csr_matrix<x_t, v_t, r_t> A(dir / "graph.bin");
            dn_matrix<r_t> X(dir / "features.bin");
            dn_matrix<std::int32_t> Y(dir / "labels.bin");
            dn_matrix<std::int32_t> S(dir / "sets.bin");

            std::cerr << A.n() << ' ' << A.nnz() << std::endl;

            const auto num_labels = 1 + *std::max_element(Y.begin(), Y.end());
            std::cerr << "num_labels = " << num_labels << std::endl;
            std::cerr << "feature size = " << X.m() << std::endl;

            auto num_sizes = std::stoi(argv[optind++]);
            std::vector<std::size_t> sizes;
            sizes.push_back(X.m());
            for (int i = 0; i < num_sizes; i++)
                sizes.push_back(std::stoull(argv[optind++]));
            sizes.push_back(num_labels);

            std::string filename;
            for (bool b = false; auto s: (dir / "graph.bin").parent_path()) {
                if (s == "permuted")
                    b = true;
                else
                    filename = (b ? std::string("permuted_") : std::string("")) + s.c_str();
            }
            
            for (auto s : sizes)
                filename += "_" + std::to_string(s);

            std::ofstream of("csvs/" + filename + "_" + std::to_string(P) + ".csv");

            if (P <= 1) {
                auto ctx = context(0);

                gcn G(A, sizes);

                ctx.sync();
                ctx.record("training-start", 0);

                for (int e = 0; e < num_epochs; e++) {
                    const auto start = std::chrono::system_clock::now();
                    auto [loss, acc] = G.train_forward(ctx, X, Y);
                    G.backward(ctx);
                    // G.update(1e-2, 5e-4);
                    G.adam_update(ctx, 1e-2, 0.9, 0.999, 5e-4, 1e-8);
                    ctx.sync();
                    const auto end = std::chrono::system_clock::now();
                    const auto duration = std::chrono::duration<double>{end - start}.count();
                    std::cerr << e << ' ' << loss << ' ' << acc << ' ' << duration << std::endl;
                    ctx.dump_timers(of, std::to_string(e) + "_0_");
                }
            }
            else if (P > 1) {
                sizes.back() = (sizes.back() + P - 1) / P * P;

                auto ctx = dist_context(P, overlap);

                std::vector<v_t> p(P + 1);
                for (std::size_t i = 1; i < p.size(); i++)
                    p[i] = i * A.n() / P;

                A.normalize(true);
                auto A_T = A.transpose();
                if (row_partition) {
                    dist_row_dn_matrix<std::int32_t> Yd(ctx, Y);

                    dist_row_csr_matrix Ad(ctx, A, p, p);
                    dist_row_csr_matrix A_Td(ctx, A_T, p, p);

                    dist_gcn<true, x_t, v_t, r_t> G(ctx, Ad, A_Td, sizes);

                    dist_row_dn_matrix Xd(ctx, X);

                    ctx.sync();
                    ctx.record("training-start", 0);
                    
                    for (int e = 0; e < num_epochs; e++) {
                        const auto start = std::chrono::system_clock::now();
                        auto [loss, acc] = G.train_forward(ctx, Xd, Yd);
                        G.backward(ctx);
                        // G.update(1e-2, 5e-4);
                        G.adam_update(ctx, 1e-2, 0.9, 0.999, 5e-4, 1e-8);
                        ctx.sync();
                        const auto end = std::chrono::system_clock::now();
                        const auto duration = std::chrono::duration<double>{end - start}.count();
                        std::cerr << e << ' ' << loss << ' ' << acc << ' ' << duration << "\n";
                        ctx.dump_timers(of, std::to_string(e) + "_");
                    }
                }
                // else {
                //     dist_csr_matrix Ad(ctx, A, p);
                //     dist_csr_matrix A_Td(ctx, A_T, p);

                //     dist_gcn<false, x_t, v_t, r_t> G(ctx, Ad, A_Td, sizes);

                //     dist_dn_matrix Xd(ctx, X);

                //     for (int i = 0; i < num_epochs; i++) {
                //         const auto start = std::chrono::system_clock::now();
                //         auto [loss, acc] = G.train_forward(ctx, Xd, Y);
                //         G.backward(ctx);
                //         // G.update(1e-2, 5e-4);
                //         G.adam_update(ctx, 1e-2, 0.9, 0.999, 5e-4, 1e-8);
                //         const auto end = std::chrono::system_clock::now();
                //         const auto duration = std::chrono::duration<double>{end - start}.count();
                //         std::cerr << i << ' ' << loss << ' ' << acc << ' ' << duration << std::endl;
                //     }
                // }
            }
        } else
            throw arg_error("Unknown command.");
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    int ret;

    try {
        ret = main_(argc, argv);
    } catch(const exception& e) {
        cerr << "Error: uncaught exception: '" << e.what() << "' Aborting." << endl;
        std::exit(EXIT_FAILURE);
    }

    return ret;
}
