---
layout: post
title:  "【C++】统计某段代码耗时"
categories: jekyll update
---

- [1. chrono header](#1-chrono-header)
- [2. 测量代码耗时示例：](#2-测量代码耗时示例)
- [3. 参考](#3-参考)

### 1. chrono header

C++ 之前没有标准的测量某段代码耗时的工具，只能依赖Boost等外部库。C++ 11 提供`chrono`头文件，可用于测量代码耗时：
- system_clock;
- high_resolution_clock;
- steady_clock

一般使用`steady_clock`，因为它能保证`t1 < t2`永远为真。

### 2. 测量代码耗时示例：

```C++
auto start = std::chrono::steady_clock::now();
//
// 要测量的代码
//
auto end = std::chrono::steady_clock::now();

// 耗时
auto diff = end - start;

// 定制打印方式
std::cout << std::chrono::duration<double, milli> (diff).count() << " ms\n";
std::cout << std::chrono::duration<double, nano> (diff).count() << " ns\n";
std::cout << std::chrono::duration_cast<std::chrono::nanoseconds> (diff).count() << " ns\n"; //取整
```

### 3. 参考

- [calculating execution time in c++](https://stackoverflow.com/questions/876901/calculating-execution-time-in-c)
- [C++11 timing code performance](https://solarianprogrammer.com/2012/10/14/cpp-11-timing-code-performance/)
