#pragma once

#if defined(LLMBRICKS_THREAD_PARALLEL)
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace llm_bricks {
class ThreadPool {
public:
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return !tasks.empty() || stop; });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task(); // 执行任务
                }
            });
        }
    }

    // 修改 enqueue，支持任意函数及其参数
    template<typename F, typename... Args>
    void enqueue(F&& func, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            // 核心思路是利用 Lambda 表达式或 std::bind，在 enqueue 时将函数及其参数绑定为一个无参的 std::function<void()> 对象，这样线程池无需改动底层逻辑
            tasks.push([func, args...]() { func(args...); });
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;
};

}   // namespace llm_bricks
#endif