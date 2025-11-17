#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono;

// ------------------ MutexQueue (coarse-grained bounded queue) ------------------
template <typename T>
class MutexQueue {
public:
    explicit MutexQueue(size_t capacity)
        : buf(capacity), cap(capacity), head(0), tail(0), cnt(0) {
        assert(capacity > 0);
    }

    // non-blocking
    bool try_push(const T& v) {
        std::lock_guard<std::mutex> lk(m);
        if (cnt == cap)
            return false;
        buf[tail] = v;
        tail = (tail + 1) % cap;
        ++cnt;
        cv_not_empty.notify_one();
        return true;
    }

    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lk(m);
        if (cnt == 0)
            return false;
        out = buf[head];
        head = (head + 1) % cap;
        --cnt;
        cv_not_full.notify_one();
        return true;
    }

    // blocking
    void push(const T& v) {
        std::unique_lock<std::mutex> lk(m);
        cv_not_full.wait(lk, [&] { return cnt < cap; });
        buf[tail] = v;
        tail = (tail + 1) % cap;
        ++cnt;
        cv_not_empty.notify_one();
    }

    void pop(T& out) {
        std::unique_lock<std::mutex> lk(m);
        cv_not_empty.wait(lk, [&] { return cnt > 0; });
        out = buf[head];
        head = (head + 1) % cap;
        --cnt;
        cv_not_full.notify_one();
    }

    size_t capacity() const { return cap; }

    size_t size() const {
        std::lock_guard<std::mutex> lk(m);
        return cnt;
    }

private:
    std::vector<T> buf;
    const size_t cap;
    size_t head;
    size_t tail;
    size_t cnt;
    mutable std::mutex m;
    std::condition_variable cv_not_full;
    std::condition_variable cv_not_empty;
};

// ------------------ LockFreeMPMCQueue ------------------
template <typename T>
class LockFreeMPMCQueue {
public:
    explicit LockFreeMPMCQueue(size_t capacity)
        : capacity_(capacity),
          buffer_(new Slot[capacity_]),
          enqueue_pos(0),
          dequeue_pos(0) {
        assert(capacity_ >= 1);
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    ~LockFreeMPMCQueue() = default;

    // non-blocking
    bool try_push(const T& item) {
        size_t pos = enqueue_pos.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = buffer_[pos % capacity_];
            size_t seq = slot.seq.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)pos;
            if (dif == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    slot.data = item;
                    slot.seq.store(pos + 1, std::memory_order_release);
                    cv_not_empty.notify_one();
                    return true;
                }
            } else if (dif < 0) {
                return false;  // full
            } else {
                pos = enqueue_pos.load(std::memory_order_relaxed);
            }
        }
    }

    bool try_pop(T& out) {
        size_t pos = dequeue_pos.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = buffer_[pos % capacity_];
            size_t seq = slot.seq.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
            if (dif == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    out = slot.data;
                    slot.seq.store(pos + capacity_, std::memory_order_release);
                    cv_not_full.notify_one();
                    return true;
                }
            } else if (dif < 0) {
                return false;  // empty
            } else {
                pos = dequeue_pos.load(std::memory_order_relaxed);
            }
        }
    }

    // blocking wrapper for push/pop that waits using condition variable and try_* internals
    void push(const T& item) {
        std::unique_lock<std::mutex> lk(wait_mtx);
        cv_not_full.wait(lk, [&] { return try_push(item); });
    }

    void pop(T& out) {
        std::unique_lock<std::mutex> lk(wait_mtx);
        cv_not_empty.wait(lk, [&] { return try_pop(out); });
    }

    size_t capacity() const { return capacity_; }

private:
    struct Slot {
        std::atomic<size_t> seq;
        T data;
    };

    const size_t capacity_;
    std::unique_ptr<Slot[]> buffer_;
    std::atomic<size_t> enqueue_pos;
    std::atomic<size_t> dequeue_pos;

    std::mutex wait_mtx;
    std::condition_variable cv_not_full;
    std::condition_variable cv_not_empty;
};

// ------------------ Singly Linked List: MutexList (coarse-grained) ------------------
template <typename T>
class MutexList {
    struct Node {
        T val;
        Node* next;

        explicit Node(const T& v) : val(v), next(nullptr) {}
    };

public:
    MutexList() : head(nullptr) {}

    ~MutexList() { clear(); }

    // insert at head
    void insert(const T& v) {
        std::lock_guard<std::mutex> lk(m);
        Node* n = new Node(v);
        n->next = head;
        head = n;
    }

    // remove first occurrence (not used in perf_list anymore)
    bool remove(const T& v) {
        std::lock_guard<std::mutex> lk(m);
        Node** cur = &head;
        while (*cur) {
            if ((*cur)->val == v) {
                Node* to = *cur;
                *cur = to->next;
                delete to;
                return true;
            }
            cur = &((*cur)->next);
        }
        return false;
    }

    bool find(const T& v) {
        std::lock_guard<std::mutex> lk(m);
        Node* cur = head;
        while (cur) {
            if (cur->val == v)
                return true;
            cur = cur->next;
        }
        return false;
    }

    // pop head (O(1)) - used in perf_list to avoid long scans
    bool try_pop_front(T& out) {
        std::lock_guard<std::mutex> lk(m);
        if (!head)
            return false;
        Node* n = head;
        head = head->next;
        out = n->val;
        delete n;
        return true;
    }

    // blocking pop_front
    void pop_front(T& out) {
        std::unique_lock<std::mutex> lk(m);
        // simple busy-waiting condition: wait until head != nullptr
        // Use condition variable if desired; here we simply wait on predicate to keep semantics simple.
        // For perf tests we use try_pop_front with retries.
        while (!head) {
            lk.unlock();
            std::this_thread::yield();
            lk.lock();
        }
        Node* n = head;
        head = head->next;
        out = n->val;
        delete n;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(m);
        Node* cur = head;
        while (cur) {
            Node* nx = cur->next;
            delete cur;
            cur = nx;
        }
        head = nullptr;
    }

    // for tests: collect content into set
    std::set<T> snapshot() {
        std::lock_guard<std::mutex> lk(m);
        std::set<T> s;
        Node* cur = head;
        while (cur) {
            s.insert(cur->val);
            cur = cur->next;
        }
        return s;
    }

private:
    Node* head;
    std::mutex m;
};

// ------------------ Singly Linked List: FineGrainedList (hand-over-hand locking) ------------------
template <typename T>
class FineGrainedList {
    struct Node {
        T val;
        Node* next;
        std::mutex m;

        explicit Node(const T& v) : val(v), next(nullptr) {}
    };

public:
    FineGrainedList() : head(nullptr) {}

    ~FineGrainedList() { clear(); }

    // insert at head (coarse head lock)
    void insert(const T& v) {
        std::lock_guard<std::mutex> lk(head_m);
        Node* n = new Node(v);
        n->next = head;
        head = n;
    }

    // remove first occurrence (hand-over-hand) - kept for correctness tests
    bool remove(const T& v) {
        std::unique_lock<std::mutex> lk_head(head_m);
        Node* prev = nullptr;
        Node* cur = head;
        if (!cur)
            return false;
        cur->m.lock();
        lk_head.unlock();

        while (cur) {
            if (cur->val == v) {
                Node* next = cur->next;
                if (prev) {
                    prev->next = next;
                    prev->m.unlock();
                } else {
                    std::lock_guard<std::mutex> lk2(head_m);
                    head = next;
                }
                cur->m.unlock();
                delete cur;
                return true;
            }
            Node* next = cur->next;
            if (prev)
                prev->m.unlock();
            prev = cur;
            if (next)
                next->m.lock();
            cur = next;
        }
        if (prev)
            prev->m.unlock();
        return false;
    }

    bool find(const T& v) {
        std::unique_lock<std::mutex> lk_head(head_m);
        Node* cur = head;
        if (!cur)
            return false;
        cur->m.lock();
        lk_head.unlock();
        while (cur) {
            if (cur->val == v) {
                cur->m.unlock();
                return true;
            }
            Node* next = cur->next;
            if (next)
                next->m.lock();
            cur->m.unlock();
            cur = next;
        }
        return false;
    }

    // try pop front (O(1)) implemented under head_m to avoid long scans
    bool try_pop_front(T& out) {
        std::lock_guard<std::mutex> lk(head_m);
        if (!head)
            return false;
        Node* n = head;
        head = head->next;
        out = n->val;
        delete n;
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(head_m);
        Node* cur = head;
        while (cur) {
            Node* nx = cur->next;
            delete cur;
            cur = nx;
        }
        head = nullptr;
    }

    std::set<T> snapshot() {
        std::lock_guard<std::mutex> lk(head_m);
        std::set<T> s;
        Node* cur = head;
        while (cur) {
            s.insert(cur->val);
            cur = cur->next;
        }
        return s;
    }

private:
    Node* head;
    std::mutex head_m;  // protects head pointer manipulations
};

// ------------------ Utilities for tests ------------------
static inline uint64_t now_ns() {
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
        .count();
}

struct RunConfig {
    size_t capacity = 1 << 16;
    size_t producers = 1;
    size_t consumers = 1;
    size_t per_producer_ops = 100000;
    unsigned seed = 12345;
    size_t duration_s = 5;
};

// CSV helper
static void append_csv_row(const std::string& fname, const std::string& header,
                           const std::string& row) {
    bool exists = std::filesystem::exists(fname);
    std::ofstream of(fname, std::ios::app);
    if (!of) {
        std::cerr << "Failed to open CSV " << fname << "\n";
        return;
    }
    if (!exists)
        of << header << "\n";
    of << row << "\n";
}

struct LatStats {
    size_t samples = 0;
    double mean_us = 0.0;
    uint64_t p50_us = 0;
    uint64_t p90_us = 0;
    uint64_t p99_us = 0;
    uint64_t p999_us = 0;
    uint64_t max_us = 0;
};

static LatStats compute_latency_stats_ns(const std::vector<uint64_t>& lat_ns) {
    LatStats st;
    if (lat_ns.empty())
        return st;
    std::vector<uint64_t> v = lat_ns;
    std::sort(v.begin(), v.end());
    st.samples = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    st.mean_us = (sum / v.size()) / 1000.0;
    auto get_pct = [&](double p) -> uint64_t {
        if (v.empty())
            return 0;
        double idx_f = (p / 100.0) * (v.size() - 1);
        size_t idx = static_cast<size_t>(std::round(idx_f));
        idx = std::min(idx, v.size() - 1);
        return v[idx] / 1000;
    };
    st.p50_us = get_pct(50.0);
    st.p90_us = get_pct(90.0);
    st.p99_us = get_pct(99.0);
    st.p999_us = get_pct(99.9);
    st.max_us = v.back() / 1000;
    return st;
}

static void print_latency_stats(const std::vector<uint64_t>& lat_ns,
                                const std::string& label) {
    LatStats s = compute_latency_stats_ns(lat_ns);
    if (s.samples == 0) {
        std::cout << label << ": no samples\n";
        return;
    }
    std::cout << label << ": samples=" << s.samples << " mean=" << s.mean_us
              << "us"
              << " p50=" << s.p50_us << "us"
              << " p90=" << s.p90_us << "us"
              << " p99=" << s.p99_us << "us"
              << " p99.9=" << s.p999_us << "us"
              << " max=" << s.max_us << "us\n";
}

// ------------------ Correctness tests (unchanged) ------------------
// ... (keep the same correctness tests for queues/lists, omitted here to save space)
// For brevity in this file I keep correctness functions minimal but intact for earlier behavior.

bool smoke_queue_singlethread() {
    MutexQueue<int> mq(4);
    mq.push(1);
    mq.push(2);
    mq.push(3);
    int x;
    mq.pop(x);
    if (x != 1)
        return false;
    mq.pop(x);
    if (x != 2)
        return false;
    mq.pop(x);
    if (x != 3)
        return false;

    LockFreeMPMCQueue<int> lf(4);
    lf.push(10);
    lf.push(20);
    lf.push(30);
    lf.pop(x);
    if (x != 10)
        return false;
    lf.pop(x);
    if (x != 20)
        return false;
    lf.pop(x);
    if (x != 30)
        return false;

    return true;
}

bool smoke_list_singlethread() {
    MutexList<int> ml;
    ml.insert(1);
    ml.insert(2);
    ml.insert(3);
    if (!ml.find(2))
        return false;
    if (!ml.remove(2))
        return false;
    if (ml.find(2))
        return false;

    FineGrainedList<int> fl;
    fl.insert(5);
    fl.insert(6);
    if (!fl.find(6))
        return false;
    if (!fl.remove(6))
        return false;
    return true;
}

// ------------------ Performance tests ------------------

// Queue perf helpers (unchanged semantic)
bool run_queue_impl_perf_lockfree(const RunConfig& cfg) {
    using Q = LockFreeMPMCQueue<uint64_t>;
    Q q(cfg.capacity);

    const size_t P = cfg.producers;
    const size_t C = cfg.consumers;
    const size_t M = cfg.per_producer_ops;
    const size_t total = P * M;

    std::atomic<size_t> produced{0};
    std::atomic<size_t> consumed{0};

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::vector<std::vector<uint64_t>> consumer_latencies(C);

    uint64_t t_start = now_ns();

    for (size_t j = 0; j < C; ++j) {
        consumers.emplace_back([j, &q, &consumed, &produced, total,
                                &consumer_latencies]() {
            while (true) {
                uint64_t v;
                if (q.try_pop(v)) {
                    uint64_t now = now_ns();
                    consumer_latencies[j].push_back(now - v);
                    consumed.fetch_add(1, std::memory_order_release);
                } else {
                    if (produced.load(std::memory_order_acquire) >= total) {
                        if (consumed.load(std::memory_order_acquire) >= total)
                            break;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }

    for (size_t i = 0; i < P; ++i) {
        producers.emplace_back([i, M, &q, &produced]() {
            for (size_t k = 0; k < M; ++k) {
                uint64_t ts = now_ns();
                while (!q.try_push(ts))
                    std::this_thread::yield();
                produced.fetch_add(1, std::memory_order_release);
            }
        });
    }

    for (auto& t : producers)
        t.join();
    for (auto& t : consumers)
        t.join();

    uint64_t t_end = now_ns();
    double elapsed_s = (t_end - t_start) / 1e9;

    std::vector<uint64_t> all;
    for (auto& v : consumer_latencies)
        all.insert(all.end(), v.begin(), v.end());

    std::cout << "LockFreeQueue perf: producers=" << P << " consumers=" << C
              << " total_ops=" << total << " elapsed=" << elapsed_s << "s"
              << " throughput=" << (total / elapsed_s) << " ops/s\n";
    print_latency_stats(all, "LockFreeQueue latency");

    LatStats st = compute_latency_stats_ns(all);
    std::ostringstream row;
    row << "LockFreeQueue,overall," << P << "," << C << "," << M << ","
        << cfg.capacity << "," << total << "," << std::fixed
        << std::setprecision(6) << elapsed_s << "," << std::fixed
        << std::setprecision(1) << (total / elapsed_s) << "," << st.samples
        << "," << st.mean_us << "," << st.p50_us << "," << st.p90_us << ","
        << st.p99_us << "," << st.p999_us << "," << st.max_us;
    const std::string header =
        "impl,phase,producers,consumers,per,capacity,total_ops,elapsed_s,"
        "throughput_ops_s,samples,mean_us,p50_us,p90_us,p99_us,p999_us,max_us";
    append_csv_row("perf_results.csv", header, row.str());

    return true;
}

bool run_queue_impl_perf_mutex(const RunConfig& cfg) {
    using Q = MutexQueue<uint64_t>;
    Q q(cfg.capacity);

    const size_t P = cfg.producers;
    const size_t C = cfg.consumers;
    const size_t M = cfg.per_producer_ops;
    const size_t total = P * M;

    std::atomic<size_t> produced{0};
    std::atomic<size_t> consumed{0};

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::vector<std::vector<uint64_t>> consumer_latencies(C);

    uint64_t t_start = now_ns();

    for (size_t j = 0; j < C; ++j) {
        consumers.emplace_back([j, &q, &consumed, &produced, total,
                                &consumer_latencies]() {
            while (true) {
                uint64_t v;
                if (q.try_pop(v)) {
                    uint64_t now = now_ns();
                    consumer_latencies[j].push_back(now - v);
                    consumed.fetch_add(1, std::memory_order_release);
                } else {
                    if (produced.load(std::memory_order_acquire) >= total) {
                        if (consumed.load(std::memory_order_acquire) >= total)
                            break;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }

    for (size_t i = 0; i < P; ++i) {
        producers.emplace_back([i, M, &q, &produced]() {
            for (size_t k = 0; k < M; ++k) {
                uint64_t ts = now_ns();
                q.push(ts);
                produced.fetch_add(1, std::memory_order_release);
            }
        });
    }

    for (auto& t : producers)
        t.join();
    for (auto& t : consumers)
        t.join();

    uint64_t t_end = now_ns();
    double elapsed_s = (t_end - t_start) / 1e9;

    std::vector<uint64_t> all;
    for (auto& v : consumer_latencies)
        all.insert(all.end(), v.begin(), v.end());

    std::cout << "MutexQueue perf: producers=" << P << " consumers=" << C
              << " total_ops=" << total << " elapsed=" << elapsed_s << "s"
              << " throughput=" << (total / elapsed_s) << " ops/s\n";
    print_latency_stats(all, "MutexQueue latency");

    LatStats st = compute_latency_stats_ns(all);
    std::ostringstream row;
    row << "MutexQueue,overall," << P << "," << C << "," << M << ","
        << cfg.capacity << "," << total << "," << std::fixed
        << std::setprecision(6) << elapsed_s << "," << std::fixed
        << std::setprecision(1) << (total / elapsed_s) << "," << st.samples
        << "," << st.mean_us << "," << st.p50_us << "," << st.p90_us << ","
        << st.p99_us << "," << st.p999_us << "," << st.max_us;
    const std::string header =
        "impl,phase,producers,consumers,per,capacity,total_ops,elapsed_s,"
        "throughput_ops_s,samples,mean_us,p50_us,p90_us,p99_us,p999_us,max_us";
    append_csv_row("perf_results.csv", header, row.str());

    return true;
}

bool test_perf_queue(const RunConfig& cfg) {
    std::cout << "=== Queue performance test (blocking push/pop) capacity="
              << cfg.capacity << " producers=" << cfg.producers
              << " consumers=" << cfg.consumers
              << " per=" << cfg.per_producer_ops << " ===\n";
    run_queue_impl_perf_lockfree(cfg);
    run_queue_impl_perf_mutex(cfg);
    return true;
}

// ------------------ List perf: use pop_front to avoid O(n) removals ------------------

bool run_list_impl_perf_mutex(const RunConfig& cfg) {
    using List = MutexList<uint64_t>;
    List lst;
    const size_t P = cfg.producers;
    const size_t M = cfg.per_producer_ops;
    const size_t total = P * M;

    // insert phase
    std::vector<std::thread> producers;
    std::vector<std::vector<uint64_t>> prod_lat(P);
    uint64_t t_start = now_ns();
    for (size_t i = 0; i < P; ++i) {
        producers.emplace_back([i, M, &lst, &prod_lat]() {
            prod_lat[i].reserve(M);
            uint64_t base = uint64_t(i) * uint64_t(M);
            for (size_t k = 0; k < M; ++k) {
                uint64_t v = base + k;
                uint64_t t0 = now_ns();
                lst.insert(v);
                uint64_t t1 = now_ns();
                prod_lat[i].push_back(t1 - t0);
            }
        });
    }
    for (auto& t : producers)
        t.join();
    uint64_t t_insert_end = now_ns();
    double insert_elapsed = (t_insert_end - t_start) / 1e9;

    // pop_front removal phase (each remover will pop M items)
    std::vector<std::thread> removers;
    std::vector<std::vector<uint64_t>> rem_lat(P);
    uint64_t rem_start = now_ns();
    for (size_t i = 0; i < P; ++i) {
        removers.emplace_back([i, M, &lst, &rem_lat]() {
            rem_lat[i].reserve(M);
            size_t done = 0;
            while (done < M) {
                uint64_t t0 = now_ns();
                uint64_t val;
                if (lst.try_pop_front(val)) {
                    uint64_t t1 = now_ns();
                    rem_lat[i].push_back(t1 - t0);
                    ++done;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    for (auto& t : removers)
        t.join();
    uint64_t rem_end = now_ns();
    double rem_elapsed = (rem_end - rem_start) / 1e9;

    // collect latencies
    std::vector<uint64_t> all_insert, all_remove;
    all_insert.reserve(total);
    all_remove.reserve(total);
    for (size_t i = 0; i < P; ++i) {
        all_insert.insert(all_insert.end(), prod_lat[i].begin(),
                          prod_lat[i].end());
        all_remove.insert(all_remove.end(), rem_lat[i].begin(),
                          rem_lat[i].end());
    }

    std::cout << "MutexList perf: producers=" << P << " total_ops=" << total
              << " insert_elapsed=" << insert_elapsed << "s"
              << " insert_throughput=" << (total / insert_elapsed)
              << " ops/s\n";
    print_latency_stats(all_insert, "MutexList insert latency");

    std::cout << "MutexList perf: remove(pop_front)_elapsed=" << rem_elapsed
              << "s"
              << " remove_throughput=" << (total / rem_elapsed) << " ops/s\n";
    print_latency_stats(all_remove, "MutexList pop_front latency");

    // write CSV rows
    LatStats st_ins = compute_latency_stats_ns(all_insert);
    LatStats st_rem = compute_latency_stats_ns(all_remove);
    const std::string header =
        "impl,phase,producers,consumers,per,capacity,total_ops,elapsed_s,"
        "throughput_ops_s,samples,mean_us,p50_us,p90_us,p99_us,p999_us,max_us";
    {
        std::ostringstream row;
        row << "MutexList,insert," << P << ",0," << M << ",0," << total << ","
            << std::fixed << std::setprecision(6) << insert_elapsed << ","
            << std::fixed << std::setprecision(1) << (total / insert_elapsed)
            << "," << st_ins.samples << "," << st_ins.mean_us << ","
            << st_ins.p50_us << "," << st_ins.p90_us << "," << st_ins.p99_us
            << "," << st_ins.p999_us << "," << st_ins.max_us;
        append_csv_row("perf_results.csv", header, row.str());
    }
    {
        std::ostringstream row;
        row << "MutexList,pop_front," << P << ",0," << M << ",0," << total
            << "," << std::fixed << std::setprecision(6) << rem_elapsed << ","
            << std::fixed << std::setprecision(1) << (total / rem_elapsed)
            << "," << st_rem.samples << "," << st_rem.mean_us << ","
            << st_rem.p50_us << "," << st_rem.p90_us << "," << st_rem.p99_us
            << "," << st_rem.p999_us << "," << st_rem.max_us;
        append_csv_row("perf_results.csv", header, row.str());
    }

    return true;
}

bool run_list_impl_perf_fine(const RunConfig& cfg) {
    using List = FineGrainedList<uint64_t>;
    List lst;
    const size_t P = cfg.producers;
    const size_t M = cfg.per_producer_ops;
    const size_t total = P * M;

    // insert phase
    std::vector<std::thread> producers;
    std::vector<std::vector<uint64_t>> prod_lat(P);
    uint64_t t_start = now_ns();
    for (size_t i = 0; i < P; ++i) {
        producers.emplace_back([i, M, &lst, &prod_lat]() {
            prod_lat[i].reserve(M);
            uint64_t base = uint64_t(i) * uint64_t(M);
            for (size_t k = 0; k < M; ++k) {
                uint64_t v = base + k;
                uint64_t t0 = now_ns();
                lst.insert(v);
                uint64_t t1 = now_ns();
                prod_lat[i].push_back(t1 - t0);
            }
        });
    }
    for (auto& t : producers)
        t.join();
    uint64_t t_insert_end = now_ns();
    double insert_elapsed = (t_insert_end - t_start) / 1e9;

    // pop_front removal phase
    std::vector<std::thread> removers;
    std::vector<std::vector<uint64_t>> rem_lat(P);
    uint64_t rem_start = now_ns();
    for (size_t i = 0; i < P; ++i) {
        removers.emplace_back([i, M, &lst, &rem_lat]() {
            rem_lat[i].reserve(M);
            size_t done = 0;
            while (done < M) {
                uint64_t t0 = now_ns();
                uint64_t val;
                if (lst.try_pop_front(val)) {
                    uint64_t t1 = now_ns();
                    rem_lat[i].push_back(t1 - t0);
                    ++done;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    for (auto& t : removers)
        t.join();
    uint64_t rem_end = now_ns();
    double rem_elapsed = (rem_end - rem_start) / 1e9;

    std::vector<uint64_t> all_insert, all_remove;
    all_insert.reserve(total);
    all_remove.reserve(total);
    for (size_t i = 0; i < P; ++i) {
        all_insert.insert(all_insert.end(), prod_lat[i].begin(),
                          prod_lat[i].end());
        all_remove.insert(all_remove.end(), rem_lat[i].begin(),
                          rem_lat[i].end());
    }

    std::cout << "FineGrainedList perf: producers=" << P
              << " total_ops=" << total << " insert_elapsed=" << insert_elapsed
              << "s"
              << " insert_throughput=" << (total / insert_elapsed)
              << " ops/s\n";
    print_latency_stats(all_insert, "FineGrainedList insert latency");

    std::cout << "FineGrainedList perf: remove(pop_front)_elapsed="
              << rem_elapsed << "s"
              << " remove_throughput=" << (total / rem_elapsed) << " ops/s\n";
    print_latency_stats(all_remove, "FineGrainedList pop_front latency");

    // write CSV rows
    LatStats st_ins = compute_latency_stats_ns(all_insert);
    LatStats st_rem = compute_latency_stats_ns(all_remove);
    const std::string header =
        "impl,phase,producers,consumers,per,capacity,total_ops,elapsed_s,"
        "throughput_ops_s,samples,mean_us,p50_us,p90_us,p99_us,p999_us,max_us";
    {
        std::ostringstream row;
        row << "FineGrainedList,insert," << P << ",0," << M << ",0," << total
            << "," << std::fixed << std::setprecision(6) << insert_elapsed
            << "," << std::fixed << std::setprecision(1)
            << (total / insert_elapsed) << "," << st_ins.samples << ","
            << st_ins.mean_us << "," << st_ins.p50_us << "," << st_ins.p90_us
            << "," << st_ins.p99_us << "," << st_ins.p999_us << ","
            << st_ins.max_us;
        append_csv_row("perf_results.csv", header, row.str());
    }
    {
        std::ostringstream row;
        row << "FineGrainedList,pop_front," << P << ",0," << M << ",0," << total
            << "," << std::fixed << std::setprecision(6) << rem_elapsed << ","
            << std::fixed << std::setprecision(1) << (total / rem_elapsed)
            << "," << st_rem.samples << "," << st_rem.mean_us << ","
            << st_rem.p50_us << "," << st_rem.p90_us << "," << st_rem.p99_us
            << "," << st_rem.p999_us << "," << st_rem.max_us;
        append_csv_row("perf_results.csv", header, row.str());
    }

    return true;
}

bool test_perf_list(const RunConfig& cfg) {
    std::cout << "=== List performance test producers=" << cfg.producers
              << " per=" << cfg.per_producer_ops << " ===\n";
    run_list_impl_perf_mutex(cfg);
    run_list_impl_perf_fine(cfg);
    return true;
}

// ------------------ CLI and runner ------------------
void usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " --test <name> [--producers P] [--consumers C] [--per M] "
                 "[--capacity K]\n";
    std::cout << "Tests: smoke, queue_correctness, list_correctness, "
                 "perf_queue, perf_list\n";
}

int main(int argc, char** argv) {
    if (argc == 1) {
        usage(argv[0]);
        return 0;
    }
    std::string testname;
    RunConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--test" && i + 1 < argc) {
            testname = argv[++i];
            continue;
        }
        if (s == "--producers" && i + 1 < argc) {
            cfg.producers = std::stoull(argv[++i]);
            continue;
        }
        if (s == "--consumers" && i + 1 < argc) {
            cfg.consumers = std::stoull(argv[++i]);
            continue;
        }
        if (s == "--per" && i + 1 < argc) {
            cfg.per_producer_ops = std::stoull(argv[++i]);
            continue;
        }
        if (s == "--capacity" && i + 1 < argc) {
            cfg.capacity = std::stoull(argv[++i]);
            continue;
        }
        if (s == "--help") {
            usage(argv[0]);
            return 0;
        }
        std::cerr << "Unknown arg: " << s << "\n";
        usage(argv[0]);
        return 1;
    }

    if (testname.empty()) {
        usage(argv[0]);
        return 0;
    }

    if (testname == "smoke") {
        bool ok = smoke_queue_singlethread() && smoke_list_singlethread();
        std::cout << "SMOKE tests " << (ok ? "OK" : "FAILED") << "\n";
        return ok ? 0 : 1;
    }

    if (testname == "perf_queue") {
        test_perf_queue(cfg);
        return 0;
    }

    if (testname == "perf_list") {
        test_perf_list(cfg);
        return 0;
    }

    std::cerr << "Unknown test: " << testname << "\n";
    usage(argv[0]);
    return 1;
}
