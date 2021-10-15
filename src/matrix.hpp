#pragma once

#include <cstdint>
#include <type_traits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cmath>
#include <random>
#include <iostream>
#include <tuple>
#include <chrono>
#include <string>
#include <mutex>
#include <iostream>
#include <map>

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>   
#include <cublas_v2.h>

#include "pigo.hpp"

#include "mg_gcn.hpp"

#include <cassert>

/** Helper class to wrap and capture argument errors */
class matrix_error : public ::std::runtime_error {
public:
    template<typename T>
        matrix_error(T t) : ::std::runtime_error(t) { }
};

const auto cusparse_create = [] (std::size_t i) {
    CHECK_CUDA( cudaSetDevice(i) );
    cusparseHandle_t handle;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    return std::shared_ptr<std::remove_pointer_t<cusparseHandle_t>>(handle, &cusparseDestroy);
};

const auto cublas_create = [] (std::size_t i) {
    CHECK_CUDA( cudaSetDevice(i) );
    cublasHandle_t handle;
    CHECK_CUBLAS( cublasCreate(&handle) );
    return std::shared_ptr<std::remove_pointer_t<cublasHandle_t>>(handle, &cublasDestroy);
};

const auto stream_create = [] (std::size_t i, int priority = 0) {
    CHECK_CUDA( cudaSetDevice(i) );
    int low, high;
    CHECK_CUDA( cudaDeviceGetStreamPriorityRange(&low, &high) );
    cudaStream_t stream;
    CHECK_CUDA( cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority == 0 ? high : low) );
    return std::shared_ptr<std::remove_pointer_t<cudaStream_t>>(stream, &cudaStreamDestroy);
};

const auto event_create = [] (std::size_t i) {
    CHECK_CUDA( cudaSetDevice(i) );
    cudaEvent_t event;
    CHECK_CUDA( cudaEventCreate(&event) );
    return std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>(event, &cudaEventDestroy);
};

class context {
    const std::size_t rank;
public:
    using timers_t = std::shared_ptr<std::map<std::string, std::pair<std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>, std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>>>>;
    
    const std::shared_ptr<std::remove_pointer_t<cusparseHandle_t>> cusparse_handle;
    const std::shared_ptr<std::remove_pointer_t<cublasHandle_t>> cublas_handle;
    const std::shared_ptr<std::remove_pointer_t<cudaStream_t>> cuda_streams[2];
    std::shared_ptr<std::map<std::string, std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>>> events;
    timers_t timers;
    std::shared_ptr<std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>>> walltimes;

    context(std::size_t index) : rank(index), cusparse_handle{cusparse_create(index)}, cublas_handle(cublas_create(index)), 
                cuda_streams{stream_create(index, 1), stream_create(index, 0)},
                events(std::make_shared<std::map<std::string, std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>>>()),
                timers(std::make_shared<std::map<std::string, std::pair<std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>, std::shared_ptr<std::remove_pointer_t<cudaEvent_t>>>>>()),
                walltimes(std::make_shared<std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>>>()) {
        CHECK_CUSPARSE( cusparseSetStream(cusparse_handle.get(), cuda_streams[0].get()) );
        CHECK_CUBLAS( cublasSetStream(cublas_handle.get(), cuda_streams->get()) );
    }

    void set() const {
        CHECK_CUDA( cudaSetDevice(rank) );
    }
    
    void sync() const {
        set();
        CHECK_CUDA( cudaDeviceSynchronize() );
	}

    auto create_event() const {
        return event_create(rank);
    }

    auto create_stream() const {
        return stream_create(rank);
    }

    void record(const std::string str, const std::size_t stream_id) {
        auto &events = *this->events;
        if (events.count(str) == 0)
            events[str] = create_event();
        CHECK_CUDA( cudaEventRecord(events[str].get(), cuda_streams[stream_id].get()) );
    }

    void wait(const std::string str, const std::size_t stream_id) {
        auto &events = *this->events;
        CHECK_CUDA( cudaStreamWaitEvent(cuda_streams[stream_id].get(), events[str].get(), 0) );
    }

    void register_timer(const std::string str, const std::string beg, const std::string end) {
        auto &timers = *this->timers;
        auto &events = *this->events;
        timers[str] = {events[beg], events[end]};
    }

    static void walltime_callback(void *time_point) {
        *reinterpret_cast<std::chrono::time_point<std::chrono::steady_clock> *>(time_point) = std::chrono::steady_clock::now();
    }

    void register_walltime(const std::string str, std::size_t stream_id) {
        walltimes->operator[](str) = std::chrono::time_point<std::chrono::steady_clock>{};
        CHECK_CUDA( cudaLaunchHostFunc(cuda_streams[stream_id].get(), walltime_callback, &walltimes->operator[](str)) );
    }

    auto measure_walltime(const std::string str) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(walltimes->operator[](str).time_since_epoch()).count();
    }

    static auto measure(timers_t timers, std::string str) {
        float ms = 0;
        auto it = timers->find(str);
        if (it != timers->end())
            CHECK_CUDA( cudaEventElapsedTime(&ms, it->second.first.get(), it->second.second.get()) );
        return ms;
    }
    
    auto measure(const std::string str) const {
        return measure(this->timers, str);
    }

    static void dump_timers(timers_t timers, std::ostream &out, std::string prefix) {
        for (auto t: *timers)
            out << prefix << t.first << ':' << measure(timers, t.first) << '\n';
    }

    void dump_timers(std::ostream &out, std::string prefix) {
        dump_timers(this->timers, out, prefix);
    }
};

std::mutex timer__iomutex;

template <typename ctx_t>
class timer {
	const std::string name;
	const std::chrono::high_resolution_clock::time_point start;
	ctx_t *ctx;
	std::mutex &mtx;
	std::ostream &os;
public:
	timer(std::string s, ctx_t *ctx = nullptr, std::mutex &_mtx = timer__iomutex, std::ostream &_os = std::cerr) : name(s), start(std::chrono::high_resolution_clock::now()), ctx(ctx), mtx(_mtx), os(_os) {
		#ifdef LOG
		std::lock_guard lock(mtx);
		os << name << " has started" << std::endl;
		#endif
	}
	~timer() {
		#ifdef LOG
		std::lock_guard lock(mtx);
		if (ctx)
			ctx->sync();
		os << name << " took " << time() * 1000 << "ms" << std::endl;
		#endif
	}
	double time() const {
		return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
	}
};

template <typename x_t, typename v_t, typename r_t>
class dist_csr_matrix;

template <typename x_t, typename v_t, typename r_t>
class dist_row_csr_matrix;

template <typename r_t>
class dist_dn_matrix;

template <typename r_t>
class dist_row_dn_matrix;

template <typename r_t>
class repl_dn_matrix;

template <typename r_t>
class dn_matrix;

/**
 * 
 * Contains matrix types here
 * 
 * More details should be added
 * 
 **/
template <typename x_t, typename v_t, typename r_t>
class csr_matrix {
private:
    v_t N_;
    v_t M_;
    cuda_ptr<x_t> indptr_;
    cuda_ptr<v_t> indices_;
    cuda_ptr<r_t> data_;
    std::shared_ptr<std::remove_pointer_t<cusparseSpMatDescr_t>> mat_;

    void read_with_PIGO(const std::filesystem::path &path) {
        pigo::CSR<v_t, v_t, v_t *, x_t *, true, r_t, r_t *> A(path.string());
        N_ = A.nrows();
        M_ = A.ncols();
        indptr_ = cuda_malloc_managed<x_t>(N_ + 1);
        std::copy(exec_policy, A.offsets(), A.offsets() + A.nrows() + 1, indptr_.get());
        indices_ = cuda_malloc_managed<v_t>(nnz());
        std::copy(exec_policy, A.endpoints(), A.endpoints() + A.m(), indices_.get());
        data_ = cuda_malloc_managed<r_t>(nnz());
        std::copy(exec_policy, A.weights(), A.weights() + A.m(), data_.get());
    }
    
    void read_binary(const std::filesystem::path &path) {
        std::ifstream ifs(path, std::ios::binary);
        const auto [v_dtype, e_dtype] = this->get_matrix_type(ifs);
        assert(v_dtype == sizeof(v_t) && e_dtype == sizeof(x_t));
        ifs.read(reinterpret_cast<char *>(&N_), sizeof(v_t));
        ifs.read(reinterpret_cast<char *>(&M_), sizeof(v_t));
        indptr_ = cuda_malloc_managed<x_t>(N_ + 1);
        ifs.read(reinterpret_cast<char *>(indptr_.get()), (N_ + 1) * sizeof(x_t));
        indices_ = cuda_malloc_managed<v_t>(nnz());
        ifs.read(reinterpret_cast<char *>(indices_.get()), nnz() * sizeof(v_t));
        data_ = cuda_malloc_managed<r_t>(nnz());
        ifs.peek();
        if (!ifs.eof())
            ifs.read(reinterpret_cast<char *>(data_.get()), nnz() * sizeof(r_t));
        else
            std::fill(data_.get(), data_.get() + nnz(), 1);
    }

    auto get_matrix_type(std::istream &in) {
        char s[4];
        in.read(s, sizeof s);
        assert(std::isdigit(s[0]));
        assert(std::isdigit(s[2]));
        s[1] = s[3] = '\0';
        return std::make_pair(std::stoi(s), std::stoi(s + 2));           
    }

public:

    auto buffer() const {
        return std::make_tuple(indptr_, indices_, data_);
    }

    void init_mat() {
        cusparseSpMatDescr_t mat;
        CHECK_CUSPARSE( cusparseCreateCsr(&mat, N_, M_, nnz(), indptr_.get(), indices_.get(), data_.get(), get_cusparse_index_type<x_t>(), get_cusparse_index_type<v_t>(), CUSPARSE_INDEX_BASE_ZERO, get_cuda_data_type<r_t>()) );
        mat_ = std::shared_ptr<std::remove_pointer_t<cusparseSpMatDescr_t>>(mat, &cusparseDestroySpMat);
    }
    
    csr_matrix() = default;

    csr_matrix(const std::filesystem::path &path) {
        if (path.extension() == ".bin") {
            // read_binary(path);
            read_with_PIGO(path);
        } else
            throw matrix_error("File type is not supported.");   
        init_mat();
    }

    csr_matrix(std::vector<x_t> indptr, std::vector<v_t> indices, std::vector<r_t> data, v_t M) : N_(indptr.size() - 1), M_(M) {
        indptr_ = cuda_malloc_managed<x_t>(indptr.size());
        std::copy(indptr.begin(), indptr.end(), indptr_.get());
        indices_ = cuda_malloc_managed<v_t>(indices.size());
        std::copy(indices.begin(), indices.end(), indices_.get());
        data_ = cuda_malloc_managed<r_t>(data.size());
        std::copy(data.begin(), data.end(), data_.get());
        init_mat();
    }

    csr_matrix(cuda_ptr<x_t> indptr, cuda_ptr<v_t> indices, cuda_ptr<r_t> data, v_t N, v_t M) : N_(N), M_(M), indptr_(indptr), indices_(indices), data_(data) {
        init_mat();
    }
    
    auto get_mat() const {
        return mat_.get();
    }

    auto n() const {
        return N_;
    }

    auto m() const {
        return M_;
    }

    auto nnz() const {
        return indptr_[N_] - indptr_[0];
    }

    auto begin(std::size_t i) const {
        return indptr_[i];
    }

    auto end(std::size_t i) const {
        return indptr_[i + 1];
    }

    auto operator[] (std::size_t i) const {
        return indices_[i];
    }

    auto as_dn() const {
        auto buffer = cuda_malloc_managed<r_t>(N_ * M_);
        std::fill(buffer.get(), buffer.get() + N_ * M_, 0);
        for (v_t v = 0; v < N_; v++) {
            for (x_t e = begin(v); e < end(v); e++) {
                buffer[v * M_ + indices_[e]] = data_[e];
            }
        }
        return dn_matrix<r_t>(N_, M_, buffer);
    }


    void normalize_par(bool axis = false) {
        if (!axis) {
            std::for_each(exec_policy, indptr_.get(), indptr_.get() + n(), [&](const auto &indptr_i) {
                const auto v = &indptr_i - indptr_.get();
                r_t sum = 0;
                for (auto e = begin(v); e < end(v); e++)
                    sum += data_[e];
                for (auto e = begin(v); e < end(v); e++)
                    data_[e] /= sum;
            });
        } else {
            auto in_degree = std::shared_ptr<r_t[]>(new r_t[M_]);
            std::fill(exec_policy, in_degree.get(), in_degree.get() + M_, 0);
            std::for_each(exec_policy, indptr_.get(), indptr_.get() + n(), [&](const auto &indptr_i) {
                const auto v = &indptr_i - indptr_.get();
                for (auto e = begin(v); e < end(v); e++)
                    in_degree[indices_[e]] += data_[e];
            });
            std::for_each(exec_policy, indptr_.get(), indptr_.get() + n(), [&](const auto &indptr_i) {
                const auto v = &indptr_i - indptr_.get();
                for (auto e = begin(v); e < end(v); e++)
                    data_[e] /= in_degree[indices_[e]];
            });
        }
    }

    void normalize(bool axis = false) {
        normalize_par(axis);
        return;
        if (!axis) {
            for (v_t v = 0; v < N_; v++) {
                r_t sum = 0;
                for (auto e = begin(v); e < end(v); e++)
                    sum += data_[e];
                for (auto e = begin(v); e < end(v); e++)
                    data_[e] /= sum;
            }
        } else {
            std::vector<r_t> in_degree(M_);
            for (v_t v = 0; v < N_; v++) {
                for (auto e = begin(v); e < end(v); e++) {
                    in_degree[indices_[e]] += data_[e];
                }
            }
            for (v_t v = 0; v < N_; v++) {
                for (auto e = begin(v); e < end(v); e++) {
                    data_[e] /= in_degree[indices_[e]];
                }
            }
        }
    }

    auto transpose_par() const {
        csr_matrix T;
        T.N_ = M_;
        T.M_ = N_;

        T.indptr_ = cuda_malloc_managed<x_t>(T.N_ + 1);
        auto indptr = std::shared_ptr<std::atomic<x_t>[]>(new std::atomic<x_t>[T.N_ + 1]);
        T.indices_ = cuda_malloc_managed<v_t>(nnz());
        T.data_ = cuda_malloc_managed<r_t>(nnz());

        std::fill(exec_policy, indptr.get(), indptr.get() + T.N_ + 1, 0);
        auto dloc = std::shared_ptr<v_t[]>(new v_t[nnz()]);
        std::for_each(exec_policy, indptr_.get(), indptr_.get() + n(), [&](const auto &indptr_i) {
            const auto i = &indptr_i - indptr_.get();
            for (auto j = begin(i); j < end(i); j++)
                dloc[j] = indptr[indices_[j] + 1].fetch_add(1, std::memory_order_relaxed);
        });

        std::copy(exec_policy, indptr.get(), indptr.get() + T.N_ + 1, T.indptr_.get());
        std::inclusive_scan(exec_policy, T.indptr_.get(), T.indptr_.get() + T.N_ + 1, T.indptr_.get());

        std::for_each(exec_policy, indptr_.get(), indptr_.get() + n(), [&](const auto &indptr_i) {
            const auto i = &indptr_i - indptr_.get();
            for (auto j = begin(i); j < end(i); j++) {
                const auto loc = T.indptr_[indices_[j]] + dloc[j];
                T.indices_[loc] = i;
                T.data_[loc] = data_[j];
            }
        });

        T.init_mat();
        return T;
    }

    auto transpose() const {
        return transpose_par();
        csr_matrix T;
        T.N_ = M_;
        T.M_ = N_;

        T.indptr_ = cuda_malloc_managed<x_t>(T.N_ + 1);
        T.indices_ = cuda_malloc_managed<v_t>(nnz());
        T.data_ = cuda_malloc_managed<r_t>(nnz());

        std::fill(T.indptr_.get(), T.indptr_.get() + T.N_ + 1, 0);
        std::vector<v_t> dloc(nnz());
        for (v_t i = 0; i < N_; i++)
            for (auto j = begin(i); j < end(i); j++)
                dloc[j] = T.indptr_[indices_[j] + 1]++;

        std::inclusive_scan(T.indptr_.get(), T.indptr_.get() + T.N_ + 1, T.indptr_.get());

        for (v_t i = 0; i < N_; i++) 
            for (auto j = begin(i); j < end(i); j++) {
                const auto loc = T.indptr_[indices_[j]] + dloc[j];
                T.indices_[loc] = i;
                T.data_[loc] = data_[j];
            }

        T.init_mat();
        return T;
    }

    auto shape() const {
        return std::make_pair((std::size_t)N_, (std::size_t)M_);
    }

    void print(std::ostream &out) const {
        for (v_t v = 0; v < N_; v++)
            for (auto e = begin(v); e < end(v); e++)
                out << "(" << v << ", " << indices_[e] << ") : " << data_[e] << '\n';
        out << std::endl << std::endl;
    }

    friend class dist_csr_matrix<x_t, v_t, r_t>;
    friend class dist_row_csr_matrix<x_t, v_t, r_t>;
};


/**
 * 
 * Contains matrix types here
 * 
 * More details should be added
 * 
 **/
template <typename r_t>
class dn_matrix {

    std::size_t N_ = 0;
    std::size_t M_ = 0;
    cuda_ptr<r_t> buffer_;
    std::shared_ptr<std::remove_pointer_t<cusparseDnMatDescr_t>> mat_;

    void read_with_PIGO(const std::filesystem::path &path) {
        pigo::ROFile f(path.string());
        N_ = f.read<std::uint32_t>();
        M_ = f.read<std::uint32_t>();
        buffer_ = cuda_malloc_managed<r_t>(N_ * M_);
        f.parallel_read(reinterpret_cast<char *>(buffer_.get()), sizeof(r_t) * N_ * M_);
    }
    
    void read_binary(const std::filesystem::path &path) {
        std::ifstream ifs(path, std::ios::binary);
        std::uint32_t N, M;
        ifs.read(reinterpret_cast<char *>(&N), sizeof(std::uint32_t));
        ifs.read(reinterpret_cast<char *>(&M), sizeof(std::uint32_t));
        N_ = N;
        M_ = M;
        buffer_ = cuda_malloc_managed<r_t>(N_ * M_);
        ifs.read(reinterpret_cast<char *>(buffer_.get()), N_ * M_ * sizeof(r_t));
    }

public:
    void init_mat() {
        cusparseDnMatDescr_t mat;
        CHECK_CUSPARSE( cusparseCreateDnMat(&mat, N_, M_, M_, buffer_.get(), get_cuda_data_type<r_t>(), CUSPARSE_ORDER_ROW) );
        mat_ = std::shared_ptr<std::remove_pointer_t<cusparseDnMatDescr_t>>(mat, &cusparseDestroyDnMat);
    }

    dn_matrix() = default;
    dn_matrix(const std::filesystem::path &path) {
        if (path.extension() == ".bin") {
            read_with_PIGO(path);
            // read_binary(path);
        } else
            throw matrix_error("File type is not supported.");    
        init_mat();          
    }

    dn_matrix(const std::size_t N, const std::size_t M) : N_(N), M_(M) {
        // @todo if we use beta != 0, set buffer to 0
        buffer_ = cuda_malloc_managed<r_t>(N_ * M_);
        init_mat();
    }

    dn_matrix(const std::pair<std::size_t, std::size_t> shape) : dn_matrix(shape.first, shape.second) {}

    // @todo check the correctness
    dn_matrix(const std::size_t N, const std::size_t M, cuda_ptr<r_t> buffer) : N_(N), M_(M), buffer_(buffer) {
        if (!buffer_)
            buffer_ = cuda_malloc_managed<r_t>(N_ * M_);
        init_mat();
    }

    dn_matrix(const std::pair<std::size_t, std::size_t> shape, cuda_ptr<r_t> buffer) : dn_matrix(shape.first, shape.second, buffer) {}

    void init(r_t gain = std::sqrt(2 / (1 + 0.01 * 0.01))) {
        std::default_random_engine gen(99);
        gain *= std::sqrt(3.0 / N_);
        std::uniform_real_distribution uni(-gain, gain);
        for (std::size_t i = 0; i < N_ * M_; i++)
            buffer_[i] = uni(gen);
    }

    void init(std::vector<r_t> f) {
        std::copy(f.begin(), f.end(), begin());
    }

    auto copy_to(const context ctx, dn_matrix<r_t> other, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
        ctx.set();
        CHECK_CUDA( cudaMemcpyAsync(other.buffer(), buffer(), size() * sizeof(r_t), copy_type, ctx.cuda_streams->get()) );
    }
    
    auto copy(const context ctx, cudaMemcpyKind copy_type = cudaMemcpyDefault) const {
        ctx.set();
        dn_matrix clone(N_, M_);
        CHECK_CUDA( cudaMemcpyAsync(clone.buffer(), buffer(), N_ * M_ * sizeof(r_t), copy_type, ctx.cuda_streams->get()) );
        return clone;
    }

    void zero(const context ctx) {
        ctx.set();    
        CHECK_CUDA( cudaMemsetAsync(buffer(), 0, N_ * M_ * sizeof(r_t), ctx.cuda_streams->get()) );
    }

    auto operator[] (std::size_t i) const {
        return buffer_[i];
    }

    auto operator[] (std::pair<std::size_t, std::size_t> p) const {
        return buffer_[p.first * M_ + p.second];
    }

    auto & operator[] (std::size_t i) {
        return buffer_[i];
    }

    auto & operator[] (std::pair<std::size_t, std::size_t> p) {
        return buffer_[p.first * M_ + p.second];
    }

    auto get_mat() const {
        return mat_.get();
    }

    auto n() const {
        return N_;
    }

    auto m() const {
        return M_;
    }

    auto size() const {
        return N_ * M_;
    }

    auto buffer() const {
        return buffer_.get();
    }

    auto shared_buffer() const {
        return buffer_;
    }

    auto begin() const {
        return buffer_.get();
    }

    auto end() const {
        return buffer_.get() + N_ * M_;
    }

    auto shape() const {
        return std::make_pair(N_, M_);
    }

    auto transpose(const context ctx) const {
        dn_matrix<r_t> mat_T(M_, N_);
        auto me = *this;
        geam(ctx, me, me, mat_T, (r_t)1, (r_t)0, true, true);
        return mat_T;
    }

    void print() const {
        for (std::size_t i = 0; i < N_; i++) {
            for (std::size_t j = 0; j < M_; j++)
                std::cout << buffer_[i * M_ + j] << ' ';
            std::cout << '\n';
        }
        std::cout << std::endl << std::endl;
    }

    friend class dist_dn_matrix<r_t>;
    friend class dist_row_dn_matrix<r_t>;
    friend class repl_dn_matrix<r_t>;
};
