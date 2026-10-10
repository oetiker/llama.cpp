// N threads, released together in a fresh process, each init their own backend on the same
// device (as libllama and whisper.cpp do) and compute a small F32 mul_mat of small integers,
// which is exact on every backend. The result is compared with the same graph on the CPU
// backend. Exit 0 if every result matches, 1 otherwise.
//
// usage: test-backend-threads [-b <device name, e.g. CPU, Vulkan0, MTL0, ROCm0>] [-t <threads>] [-i <iterations>]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static const int M = 16, N = 12, K = 32;

// One fresh backend on dev, one graph, one compute. Returns false on any failure; out holds the result.
static bool compute_once(ggml_backend_dev_t dev, std::vector<float> & out) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_dev_init failed\n", ggml_backend_dev_name(dev));
        return false;
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * prod = ggml_mul_mat(ctx, a, b);
    ggml_set_output(prod);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, prod);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(galloc, gf);
    std::vector<float> va(M * K), vb(N * K);
    for (int i = 0; i < M * K; i++) { va[i] = (float)(i % 7); }
    for (int i = 0; i < N * K; i++) { vb[i] = (float)(i % 5); }
    if (ok) {
        ggml_backend_tensor_set(a, va.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, vb.data(), 0, ggml_nbytes(b));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.resize(M * N);
        ggml_backend_tensor_get(prod, out.data(), 0, ggml_nbytes(prod));
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

// Releases all waiting threads at once, so they race into ggml_backend_dev_init together.
struct start_gate {
    std::mutex mtx;
    std::condition_variable cv;
    int waiting = 0;
    int count;

    explicit start_gate(int n) : count(n) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        if (++waiting == count) {
            cv.notify_all();
        } else {
            cv.wait(lock, [this] { return waiting == count; });
        }
    }
};

int main(int argc, char ** argv) {
    const char * want_dev = nullptr;
    int threads = 4, iterations = 16;
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc && !strcmp(argv[i], "-b")) {
            want_dev = argv[++i];
        } else if (i + 1 < argc && !strcmp(argv[i], "-t")) {
            threads = atoi(argv[++i]);
        } else if (i + 1 < argc && !strcmp(argv[i], "-i")) {
            iterations = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [-b <device name>] [-t <threads>] [-i <iterations>]\n", argv[0]);
            return 1;
        }
    }
    if (threads < 1 || iterations < 1) {
        fprintf(stderr, "threads and iterations must be >= 1\n");
        return 1;
    }

    ggml_backend_load_all();

    // The reference comes from the CPU device. Computing it does not touch any other device,
    // so the first init of the device under test still happens in the racing threads.
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    std::vector<float> want;
    if (!cpu || !compute_once(cpu, want)) {
        printf("CPU reference failed\n");
        return 1;
    }

    int failures = 0, tested = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); d++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (want_dev && strcmp(ggml_backend_dev_name(dev), want_dev) != 0) { continue; }
        tested++;
        std::atomic<int> bad{0};
        start_gate gate(threads);
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&, t] {
                gate.arrive_and_wait();
                for (int it = 0; it < iterations; it++) {
                    std::vector<float> got;
                    if (!compute_once(dev, got)) {
                        bad++;
                        fprintf(stderr, "%s thread %d iteration %d: compute failed\n", ggml_backend_dev_name(dev), t, it);
                    } else if (got != want) {
                        bad++;
                        fprintf(stderr, "%s thread %d iteration %d: result mismatch\n", ggml_backend_dev_name(dev), t, it);
                    }
                }
            });
        }
        for (auto & th : pool) { th.join(); }
        printf("%s (%s): %d threads x %d: %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               threads, iterations, bad ? "FAIL" : "OK");
        failures += bad > 0;
    }
    if (tested == 0) {
        printf("no device matched%s%s\n", want_dev ? ": " : "", want_dev ? want_dev : "");
        return 1;
    }
    return failures ? 1 : 0;
}
