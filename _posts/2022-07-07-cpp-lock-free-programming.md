---
layout: post
title:  "【C++】chaonet: 一个轻量网络编程库"
categories: jekyll update
---

### 无锁编程

所谓无锁编程，就是用来编写不显示使用锁的并发程序技术。程序员转而依靠由硬件支持的**原语操作**来避免数据竞争。这里的原语操作在C++标准库里值得就是使用原子类型的**原子操作**，同时相较传统的原语操作，C++标准提供的原子操作是可移植的。原子操作的背后当然是由C++提供的内存模型支持，详细可[我之前的博客]()。

本文主要介绍如何利用C++标准提供的原子类型进行无锁编程，实现两个常用的数据结构，它们不仅是线程安全的，而且较有锁编程，无锁编程最大限度提升并发度，这主要得益于无锁编程：

- 非阻塞，甚至可能是免等的。总是存在某个线程能执行下一步操作；

- 利用特定内存次序做特定的操作。

无锁编程的另外一个优点是**代码健壮性**，因为较有锁编程，无锁编程代码即便在某线程意外终止，也仅仅影响其持有的数据对其他数据无影响。

然而，无锁编程也有许多缺点：

- 编写难度大。必须格外谨慎;

- 能够避免死锁，但是可能出现**活锁**，即线程间执行到某一处都需要更改同一数据，于是反复循环，又同时到达该处，如此反复。这种情况非常少见，但仍然可能降低代码效率；

- 同步数据时可能缓存乒乓。即多个独立CPU核心频繁轮流读写同一数据，使各个CPU所属的缓存不断地切如、切出（回写），导致严重性能问题。

下面利用无锁编程实现一个栈和一个队列，尽量体现无锁编程的优点，避免其缺点。

### 实现线程安全的栈

用无锁编程实现栈需要用到特别多的trick：

- 使用`compare_and_exchange`。这没得说，使用原子类型的`compare_and_exchange`函数检查是否有其他线程正在修改操作的对象，如果修改了，则改变为修改的值，再进行比较，直到没有修改，就给当前对象赋值；

- 使用内外两个引用计数。非常巧妙的内存管理方式。在弹栈时，最大的困难就是防止内存泄露，因为即便弹出栈了，我们也不知道还有没有其他线程在读取该节点，也就不知道何时该删除次节点。因此必须自己实现一个类似垃圾回收的机制。

- 内存次序。为了最大限度提升并发，在完成所有功能后，我们要尽可能地放宽原子操作的内存次序，当然前提是保证代码功能符合预期。这就需要对代码间的关系进行分析。



### 实现线程安全的队列

用无锁编程实现队列和上面实现锁的方式类似，关键也是使用两个引用计数来做内存管理，代码如下：

```c++

//
// an implementation of a lock free queue. It is similar to the implementation of the lock free stack. Both of them use
// external and internal reference count to manage memory.
//

#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueue {
private:
    struct Node;
    struct CountedNodePtr {
        int externalCount;
        Node* ptr;
    };
    std::atomic<CountedNodePtr> head;
    std::atomic<CountedNodePtr> tail;

    struct NodeCounter {
        unsigned internalCount:30;
        unsigned externalCounters:2;
    };
    struct Node {

        // decrease reference count
        void releaseRef() {
            NodeCounter oldCounter = count.load(std::memory_order_relaxed);
            NodeCounter newCounter;
            do {
                newCounter = oldCounter;
                --newCounter.internalCount;
            } while (!count.compare_exchange_strong(oldCounter, newCounter, std::memory_order_acquire, std::memory_order_relaxed));
            if (!newCounter.internalCount && !newCounter.externalCounters) {
                delete this;
            }
        }

        std::atomic<T*> data;
        std::atomic<NodeCounter> count;
        CountedNodePtr next;
        Node() {
            NodeCounter newCount;
            newCount.internalCount = 0;
            newCount.externalCounters = 2;
            count.store(newCount);
            next.ptr = nullptr;
            next.externalCount = 0;
        }
    };

    // increase reference count of counter
    static void increaseExternalCount(std::atomic<CountedNodePtr>& counter, CountedNodePtr& oldCounter) {
        CountedNodePtr newCounter;
        do {
            newCounter = oldCounter;
            ++newCounter.externalCount;
        } while (!counter.compare_exchange_strong(oldCounter, newCounter, std::memory_order_acquire, std::memory_order_relaxed));
        oldCounter.externalCount = newCounter.externalCount;
    }

    // add external reference count to internal, delete external counter if it decreases to zero.
    static void freeExternalCounter(CountedNodePtr& oldNodePtr) {
        Node* const ptr = oldNodePtr.ptr;
        int const countIncrease = oldNodePtr.externalCount - 2;
        NodeCounter oldCounter = ptr->count.load(std::memory_order_relaxed);
        NodeCounter newCounter;
        do {
            newCounter = oldCounter;
            --newCounter.externalCounters;
            newCounter.internalCount += countIncrease;
        } while (!ptr->count.compare_exchange_strong(oldCounter, newCounter, std::memory_order_acquire, std::memory_order_relaxed));
        if (!newCounter.internalCount && !newCounter.externalCounters) {
            delete ptr;
        }
    }

public:
    LockFreeQueue () {
        CountedNodePtr newNext; 
        newNext.ptr = new Node;
        newNext.externalCount = 1;
        head.store(newNext);
        tail.store(head);
    }

    void push (T newValue) {
        auto newData = std::make_unique<T>(newValue);
        CountedNodePtr newNext;
        newNext.ptr = new Node;
        newNext.externalCount = 1;
        CountedNodePtr oldTail = tail.load();
        for (;;) {
            increaseExternalCount(tail, oldTail);
            T* oldData = nullptr;

            // push data to the dummy tail node, the make newNext be new dummy tail
            if (oldTail.ptr->data.compare_exchange_strong(oldData, newData.get())) {
                oldTail.ptr->next = newNext;
                oldTail = tail.exchange(newNext);
                freeExternalCounter(oldTail);
                newData.release();
                break;
            }
            oldTail.ptr->releaseRef();
        }
    }

    std::unique_ptr<T> pop() {
        CountedNodePtr oldHead = head.load(std::memory_order_relaxed);
        for (;;) {
            increaseExternalCount(head, oldHead);
            Node* const ptr = oldHead.ptr;

            // this means queue has only a dummy tail. It is empty.
            if (ptr == tail.load().ptr) {
                ptr->releaseRef();
                return std::unique_ptr<T>();
            }

            // exchange head to head->next
            if (head.compare_exchange_strong(oldHead, ptr->next)) {
                T* const res = ptr->data.exchange(nullptr);
                freeExternalCounter(oldHead);
                return std::unique_ptr<T>(res);
            }
            ptr->releaseRef();
        }
    }
};
```

### 参考文献

- [C++并发编程实战](https://book.douban.com/subject/26)
