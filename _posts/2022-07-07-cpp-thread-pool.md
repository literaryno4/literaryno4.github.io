---
layout: post
title:  "【C++并发编程】如何写出一个高效的线程池"
categories: jekyll update
---

### 引言

要写一个线程池贼简单，只需要一个任务队列，然后用互斥锁保证该队列线程安全，然后创建若干线程，从任务队列执行就OK了。比如这样写：
```c++
class ThreadPoolLock {
    bool finished;
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> threads_;
    std::mutex mtx;
public:
    ThreadPoolLock() : finished(false) {
        int numCore = std::thread::hardware_concurrency();
        int numThread = std::min(numCore == 0 ? 2 : numCore, 8);
        for (int i = 0; i < numThread; ++i) {
            threads_.push_back(std::thread(&ThreadPoolLock::threadFuc, this));
        }
    }
    ~ThreadPoolLock() {
        finished = true;
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    template <typename Func>
    void submitTask(Func f) {
        std::lock_guard<std::mutex> lk(mtx);
        tasks.push(std::function<void()>(f));
    }

    void threadFuc() {
        while (!finished) {
            std::unique_lock<std::mutex> lk(mtx);
            if (!tasks.empty()) {
                auto task = tasks.front();
                tasks.pop();
                lk.unlock();
                task();
            } else {
                lk.unlock();
                std::this_thread::yield();
            }
        }
    }
};
```
40行代码就搞定了。也可以用之前博客的[线程安全的队列](https://literaryno4.github.io/threadsafe-data-struct.html/)，更加简单：
```c++
class ThreadPool {
    bool finished;
    LockFreeQueue<std::function<void()>> tasks;
    std::vector<std::thread> threads_;
public:
    ThreadPool() : finished(false) {
        int numCore = std::thread::hardware_concurrency();
        int numThread = std::min(numCore == 0 ? 2 : numCore, 8);
        for (int i = 0; i < numThread; ++i) {
            threads_.push_back(std::thread(&ThreadPool::threadFuc, this));
        }
    }
    ~ThreadPool() {
        finished = true;
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    template <typename Func>
    void submitTask(Func f) {
        tasks.push(std::function<void()>(f));
    }

    void threadFuc() {
        while (!finished) {
            auto taskPtr = tasks.pop();
            if (taskPtr) {
                std::function<void()> task = *taskPtr;
                task();
            } else {
                std::this_thread::yield();
            }
        }
    }
};
```
看起来是不是很简单，这样的线程池的确能工作，比如下面的场景，实现并发地给一个数组的每个元素加一：
```c++
int main() {
    int numOfV = 134;
    std::vector<int> v(numOfV);
    for (int i = 0; i < numOfV; ++i) {
        v[i] = i;
    }

    int length = v.size();

    ThreadPool tp;
    int numPerThread = 25;
    int i;
    for (i = 0; i <= length - numPerThread; i += numPerThread) {
        tp.submitTask([=, &v]() {
                          Task()(v, i, i + numPerThread - 1);
                      });
        printf("%d\n", i);
    }
    Task()(v, i, length - 1);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<int> v2(numOfV);
    for (int i = 0; i < numOfV; ++i) {
        v2[i] = i + 1;
    }
    assert(v == v2);
}
```

这段代码确实能工作，也实实在在会并发运行，但是我们明显能看出以前奇怪的，不自然的地方：

- 为什么主线程要sleep？因为我们要保证其他队列里的任务都完成。那数据如果很大，该sleep多久呢？这是一个很大的问题。应该找到另外的方式来确保队列任务都已完成。

- 上面的线程池只有一个全局的任务队列，如果线程数目很大，必然发生激烈的任务争夺。即便使用无锁编程实现的队列避开显式等待，也会产生缓存乒乓现象。因此必须想办法避免任务争夺。

对于第一个问题，解决的办法很多，比如添加计数器，在析构的时候进行检查。但是更通用的方式是通过`future`和`promise`，因为它不仅能够检查任务是否完成，还能够传递数据。

对于第二个问题，可以每个线程都拥有独立的任务队列，一般情况下线程都是从自己线程的队列中获取任务，这样就能有效的避免数据竞争或者缓存乒乓。事实上，[chaonet](https://literaryno4.github.io/chaonet-overview.html/)的线程池就是这样做的，每个线程（EventLoop）都有自己独立的任务队列。

下面针对以上问题对线程池进行优化，为了方便，直接使用之前实现的线程安全的队列。

### 可以等待任务完成的线程池

主要通过[future 和 promise](https://literaryno4.github.io/cpp-concurrency-interface.html/)来进行主线程对其他线程的同步。具体的，在主线程获取每个任务的future，然后利用`future.get()`显式等待所有任务完成。线程池代码如下：

```c++
class ThreadPool2 {
    bool finished;
    const int maxThread = 25;
    LockFreeQueue<std::packaged_task<void()>> tasks;
    std::vector<std::thread> threads_;
public:
    ThreadPool2() : finished(false) {
        int numCore = std::thread::hardware_concurrency();
        int numThread = std::min(numCore == 0 ? 2 : numCore, maxThread);
        for (int i = 0; i < numThread; ++i) {
            threads_.push_back(std::thread(&ThreadPool2::threadFuc, this));
        }
    }

    ~ThreadPool2() {
        finished = true;
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::vector<std::thread>& getThreads() {
        return threads_;
    }
    int sizeOfThreads() {
        return threads_.size();
    }

    // when submit task, return a future to the productor for communication.
    template <typename Func>
    std::future<void> submitTask(Func f) {
        std::packaged_task<void()> task(std::move(f));
        std::future<void> res(task.get_future());
        tasks.push(std::move(task));
        return res;
    }

    void threadFuc() {
        while (!finished) {
            auto taskPtr = tasks.pop();
            if (taskPtr) {
                auto& task = *taskPtr;
                task();
            } else {
                std::this_thread::yield();
            }
        }
    }
};
```
测试代码如下：
```c++
int main() {
    int numOfV = 13400000;
    std::vector<int> v(numOfV);
    for (int i = 0; i < numOfV; ++i) {
        v[i] = i;
    }

    int length = v.size();

    ThreadPool2 tp;
    int numPerThread = 25;
    std::vector<std::future<void>> futs(length / numPerThread);
    int i, idx;

    // divide task and dispatch to thread pool.
    for (i = 0, idx = 0; i <= length - numPerThread; i += numPerThread, ++idx) {
        futs[idx] = tp.submitTask([=, &v]() {
                          Task()(v, i, i + numPerThread - 1);
                      });
    }
    Task()(v, i, length - 1);

    for (auto& fut : futs) {
        fut.get();
    }

    std::vector<int> v2(numOfV);
    for (int i = 0; i < numOfV; ++i) {
        v2[i] = i + 1;
    }

    assert(v == v2);
}
```
可以看到，利用`future`可以及时等待线程池所有任务完成，然后继续后续任务。

### 避免任务队列上的争夺

在引言中说到，如果线程池共享一个任务队列，会造成数据竞争或者缓存乒乓，损失性能。可以通过给每个线程分配一个独立的任务队列，在提交任务的时候轮流添加到每个任务队列中。这样由于每个任务队列相互独立，即便使用线程不安全的队列，也仅需很少的同步工作，利用`std::list`做任务队列的实现如下：
```c++
template <typename Func, typename T>
class ThreadPool3 {
    using QueueType = std::list<std::packaged_task<Func>>;
    const int maxThread_ = 25;
    bool done_;
    int queueIdx_;
    std::mutex mtx_;
    std::vector<std::thread> threads_;
    std::vector<QueueType> queues_;

    QueueType& getNextQueue() {
        int idx = queueIdx_;
        ++queueIdx_;
        if (queueIdx_ == queues_.size()) {
            queueIdx_ = 0;
        }
        return queues_[idx];
    }

public:
    using resultType = T;

    ThreadPool3 () : queueIdx_(0), done_(false){
        int numCore = std::thread::hardware_concurrency();
        int numThread = std::min(numCore == 0 ? 2 : numCore, maxThread_);
        for (int i = 0; i < numThread; ++i) {
            queues_.push_back(std::list<std::packaged_task<Func>>());
        }
        for (int i = 0; i < numThread; ++i) {
            threads_.push_back(std::thread(&ThreadPool3::threadFunc, this, i));
        }
    }

    ~ThreadPool3() {
        for (auto& q : queues_) {
            while (q.size());
        }
        done_ = true;
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void threadFunc(int queueIdx) {
        auto& q = queues_[queueIdx];
        while (!done_) {
            if (q.size()) {
                auto& task = q.front();
                task();
                std::unique_lock<std::mutex> lk(mtx_);
                q.pop_front();
                lk.unlock();
            } else {
                std::this_thread::yield();
            }
        }
    }

    template <typename F>
    std::future<resultType> submitTask(F f) {
        std::packaged_task<Func> task(std::move(f));
        std::future<resultType> res(task.get_future());
        auto& q = getNextQueue();
        std::unique_lock<std::mutex> lk(mtx_);
        q.push_back(std::move(task));
        lk.unlock();
        return res;
    }
};
```
这一次，在创建线程的时候，通过任务函数参数区分任务队列，即第i个线程对应第i个任务队列。在提交任务的时候，通过`getNextQueue()`函数轮流提交到各个线程的任务队列上。注意在提交任务的时候和弹出任务的时候需要加锁。使用线程安全的队列可以避免使用锁，提升效率：
```c++
template <typename Func, typename ReturnType>
class ThreadPool4{
    using TaskType = std::packaged_task<Func>;
    using QueueType = LockFreeQueue<TaskType>;

    bool done_ = false;
    int queueIdx_;
    const int maxThread = 25;
    const int numCores = std::thread::hardware_concurrency();
    const int numThread = std::min(numCores == 0 ? 2 : numCores, maxThread);
    std::vector<std::thread> threads_;
    std::vector<QueueType> queues_;

    QueueType& getNextQueue() {
        int idx = queueIdx_;
        ++queueIdx_;
        if (queueIdx_ == numThread) {
            queueIdx_ = 0;
        }
        return queues_[idx];
    }

public:
    ThreadPool4() : queues_(numThread) {
        for (int i = 0; i < numThread; ++i) {
            threads_.emplace_back(&ThreadPool4::threadFunc, this, i);
        }
    }
    ~ThreadPool4() {
        done_ = true;
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void threadFunc(int queueIdx) {
        while (!done_) {
            auto& q = queues_[queueIdx];
            auto taskPtr = q.pop();
            if (taskPtr) {
                (*taskPtr)();
            } else {
                std::this_thread::yield();
            }
        }
    } 

    template <typename F>
    std::future<ReturnType> submitTask(F f) {
        std::packaged_task<Func> task(std::move(f));
        auto res = task.get_future();
        auto& q = getNextQueue();
        q.push(std::move(task));

        return res;
    }
};

```

### 总结

通过分析简单的线程池的问题，利用future和单独任务队列解决这些问题，从而实现了一个较为通用和高效的线程池。具体的性测试代码可以在[这里](https://github.com/literaryno4/datastructalgo)。

### 参考文献

- [C++并发编程实战](https://book.douban.com/subject/26)
