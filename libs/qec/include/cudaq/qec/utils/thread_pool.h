/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace cudaq::qec::utils {

class ThreadPool {
public:
    // Option 1: Standard unpinned thread pool
    explicit ThreadPool(size_t threads);

    // Option 2: Pinned thread pool (1 thread per specified core ID)
    explicit ThreadPool(const std::vector<int>& core_ids);

    ~ThreadPool();

    // Enqueue a job into the pool.
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    void worker_loop();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// --- Implementation ---

inline void ThreadPool::worker_loop() {
    while(true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this] {
                return this->stop || !this->tasks.empty();
            });

            if(this->stop && this->tasks.empty()) {
                return;
            }

            task = std::move(this->tasks.front());
            this->tasks.pop();
        }
        task();
    }
}

// Constructor 1: Unpinned
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] { this->worker_loop(); });
    }
}

// Constructor 2: Pinned to specific cores
inline ThreadPool::ThreadPool(const std::vector<int>& core_ids) : stop(false) {
    for(size_t i = 0; i < core_ids.size(); ++i) {
        int core_id = core_ids[i];

        workers.emplace_back([this, core_id] {
            // Apply Thread Affinity (Linux Only)
#if defined(__linux__)
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core_id, &cpuset);

            int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) {
                std::cerr << "[ThreadPool] Warning: Failed to pin thread to core "
                          << core_id << " (Error " << rc << ")\n";
            }
#else
            // Silent fallback for non-Linux platforms
            (void)core_id;
#endif

            // Enter the standard execution loop
            this->worker_loop();
        });
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace cudaq::qec::utils
