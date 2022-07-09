---
layout: post
title:  "【C++】实现线程安全的数据结构"
categories: jekyll update
---

<!-- vim-markdown-toc GFM -->

* [引言](#引言)
* [线程安全、异常安全且高并发度的队列](#线程安全异常安全且高并发度的队列)
* [线程安全、异常安全且高并发度的链表](#线程安全异常安全且高并发度的链表)
* [参考文献](#参考文献)

<!-- vim-markdown-toc -->

### 引言
C++ STL 提供的数据结构基本都不是线程安全的，包括vector、list、map、unordered_map等常用容器。这意味着编写多线程程序时，必须依靠用户来保证容器使用的线程安全性。然而，用户是很难做到的，原因如下：
- 标准库容器大多数都十分依赖迭代器，如果按照单线程的习惯使用容器，将十分容易造成迭代器失效的灾难。例如，一个线程持有一个list的迭代器，另一个线程对该迭代器指向的节点执行删除操作，当之前的线程再次使用其持有的迭代器时，它已经失效；
- 退一万步讲，就算用户好不容易保证了使用容器的相关代码都线程安全，使用多线程的意义已然不复存在。因为保证标准容器线程安全必然使用大粒度的锁，例如给容器每个成员函数都加锁，这样造成的结果就是程序的并发度荡然无存。

我们必须尽量减小容器操作的锁粒度，从而保证并发度。例如，当一个`stack`压栈的时候，没有必要在创建栈元素的时候就直接加上锁，实际上只需要在元素创建完成之后，对栈成员变量进行修改的时候加锁。这显然需要我们从容器内部修改其代码实现，而不是简单地对容器提供的接口进行加锁。

下面主要以实现线程安全的队列和链表为例进行介绍。

### 线程安全、异常安全且高并发度的队列

实现一个队列非常简单，只需要保存记录头节点和尾节点的两个指针`head`和`tail`，然后根据出队和入队操作更新这两个指针即可。但在实现线程安全的队列的时候，为了降低锁的粒度，实现高并发，应该注意一些技巧：
- **分离数据**。为了降低队列操作的持锁时间，应该使用两个互斥锁分别对`head`和`tail`进行保护，但这会增加实现的复杂度，我们也应该更加谨慎，避免出现竞争和死锁。使用一个虚节点可以保证在入队操作的时候，避免访问`head`指针（不加的话，当队列为空时，我们不得不访问`head`），我们只需要对`tail`指针进行加锁即可。
- 等待数据弹出。如果队列为空时进行出队操作，我们不应该仅仅抛出异常就万事大吉，应提供一个等待弹出的接口，即当队列为空的时候等待队列不空后再出队，很显然这里应该使用条件变量。等待数据弹出这一点在实际应用中非常有意义，因为谁也无法保证生产者和消费者对队列的操作速度完美匹配。

实现代码如下：

```c++
#include <memory>
#include <mutex>
#include <condition_variable>

namespace structalgo {

template <typename T>
class ThreadsafeQueue {
    struct Node {
        std::shared_ptr<T> data;
        std::unique_ptr<Node> next;
        Node() : next(nullptr) {}
    };
    std::unique_ptr<Node> head_;
    std::mutex headMutex_;
    Node* tail_;
    std::mutex tailMutex_;
    std::condition_variable dataCondVar_;

    // helper functions
    Node* getTail() {
        std::lock_guard<std::mutex> lk(tailMutex_);
        return tail_;
    }

    std::unique_ptr<Node> popHead() {
        std::unique_ptr<Node> oldHead = std::move(head_);
        head_ = std::move(oldHead->next);
        return oldHead;
    }

    std::unique_lock<std::mutex> waitForData() {
        std::unique_lock<std::mutex> lk(headMutex_);
        dataCondVar_.wait(lk, [&]() { return head_.get() != getTail(); });
        return std::move(lk);
    }

    std::unique_ptr<Node> waitPopHead() {
        std::unique_lock<std::mutex> lk(waitForData());
        return popHead();
    }

    std::unique_ptr<Node> waitPopHead(T& value) {
        std::unique_lock<std::mutex> lk(waitForData());
        value = std::move(*head_->data);
        return popHead();
    }

public:
    ThreadsafeQueue() : head_(new Node), tail_(head_.get()) {}
    bool empty() {
        std::lock_guard<std::mutex> lk(headMutex_);
        return head_.get() == getTail();
    }

    void push(T data) {
        auto newData = std::make_shared<T>(std::move(data));
        auto newNode = std::make_unique<Node>();
        Node* newTail = newNode.get();
        {
            std::lock_guard<std::mutex> lk(tailMutex_);
            tail_->data = std::move(newData);
            tail_->next = std::move(newNode);
            tail_ = newTail;
        }
        dataCondVar_.notify_one();
    }

    std::shared_ptr<Node> tryPop() {
        std::lock_guard<std::mutex> lk(headMutex_);
        if (head_.get() == getTail()) {
            return std::shared_ptr<Node>();
        }
        return popHead();
    }

    std::unique_ptr<Node> tryPop(T& value) {
        std::lock_guard<std::mutex> lk(headMutex_);
        if (head_.get() == getTail()) {
            return std::shared_ptr<T>();
        }
        std::unique_ptr<Node> oldHead = std::move(popHead());
        value = *(oldHead->data);
        return oldHead;
    }

    std::shared_ptr<T> waitAndPop() {
        std::unique_ptr<Node> oldHead = waitPopHead();
        return oldHead->data;
    }

    void waitAndPop(T& value) {
        std::unique_ptr<Node> oldHead = waitPopHead(value);
    }
};

}
```
分析代码可以知道，除了线程安全，程序的异常安全可由`std::shared_ptr`保证。

### 线程安全、异常安全且高并发度的链表

要实现线程安全且高并发的链表同样需要考虑许多细节：
- 每个节点独立拥有互斥量。这样做的好处是，当访问到链表的某个节点时，只对当前节点加锁，其他线程可以继续访问当前节点之前的节点，无需等待；

- 在从当前节点到下一个节点时，应该先锁住下一个节点，然后在解锁当前节点，这样才能保证线程安全。

一个简单的单向链表实现如下：

```c++
template <typename T>
class ThreadsafeList {
    //
    // every node has a unique mutex, which make threads visit list
    // concurrently. note this method must keep threads visiting list
    // sequentially. For example: list: A->B->C->D->E. if thread1 is visiting
    // node C, new threads can only visit nodes before node C
    //
    struct Node {
        std::mutex m;
        std::shared_ptr<T> data;
        std::unique_ptr<Node> next;
        Node() : next() {}
        Node(T&& value) : data(std::make_shared<T>(std::move(value))), next() {}
    };

    // this forward list has a dummy head
    Node head;

   public:
    ThreadsafeList() {}
    ~ThreadsafeList() {
        removeIf([](auto&){ return true; });
    }

    ThreadsafeList(const ThreadsafeList&) = delete;
    ThreadsafeList& operator=(const ThreadsafeList&) = delete;

    // add new node after the dummy head
    void pushFront(T&& data) {
        auto newHead = std::make_unique<Node>(std::move(data));
        std::lock_guard<std::mutex> lk(head.m);
        newHead->next = std::move(head.next);
        head.next = std::move(newHead);
    }

    void insertAfter(Node* pos, T&& data) {
        auto newNode = std::make_unique<Node>(std::move(data));
        std::unique_lock<std::mutex> lk(head.m);
        Node* current = &head;
        while ((current = current->next.get())) {
            std::unique_lock<std::mutex> lk_next(current->m);
            lk.unlock();
            if (current == pos) {
                newNode->next = std::move(current->next);
                current->next = std::move(newNode);
            }
            lk = std::move(lk_next);
        }
    }

    //
    // Iterate this list and do something to every node.
    // to safely iterate, we must make other thread no way to pass by current node
    // that is to say:
    // list: A->B->C->D->E
    // first lock Node B, then make sure lock C before unlock B
    //
    template <typename Func>
    void forEach(Func func) {
        Node* current = &head;
        std::unique_lock<std::mutex> lk(head.m);
        while ((current = current->next.get())) {
            std::unique_lock<std::mutex> lk_next(current->m);
            lk.unlock();
            func(*current->data);
            lk = std::move(lk_next);
        }
    }

    //
    // find first of node which satisfy pred
    // to safely iterate, we must make other thread no way to pass by current node
    // that is to say:
    // list: A->B->C->D->E
    // first lock Node B, then make sure lock C before unlock B
    //
    template <typename Pred>
    Node* findFirstIF(Pred pred) {
        Node* current = &head;
        std::unique_lock<std::mutex> lk(head.m);
        while ((current = current->next.get())) {
            std::unique_lock<std::mutex> lk_next(current->m);
            lk.unlock();
            if (pred(*current->data)) {
                return current;
            }
            lk = std::move(lk_next);
        }
        return nullptr;
    }

    // remove first of node which satisfy pred
    template <typename Pred>
    void removeIf(Pred pred) {
        Node* current = &head;
        std::unique_lock<std::mutex> lk(head.m);
        while (current->next) {
            std::unique_lock<std::mutex> lk_next(current->next->m);
            if (pred(*current->next->data)) {
                auto oldNext = std::move(current->next);
                current->next = std::move(oldNext->next);
                lk_next.unlock();
            } else {
                current = current->next.get();
                lk = std::move(lk_next);
            }
        }
    }

    void remove(const T& value) {
        removeIf([&](auto& data) {return data == value; });
    }
};
```

### 参考文献

- [C++并发编程实战](https://book.douban.com/subject/26)
