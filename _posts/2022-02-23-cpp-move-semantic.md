---
layout: post
title:  "【C++】移动语义（std::move）"
categories: jekyll update
---

- [1. `std::move`](#1-stdmove)
- [2. 移动语义的本质](#2-移动语义的本质)
- [3. 测试](#3-测试)

### 1. `std::move`

`std::move`的作用是将参数转化为右值。其实现很简单：
```c++
template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR
typename remove_reference<_Tp>::type&&
move(_Tp&& __t) _NOEXCEPT
{
    typedef typename remove_reference<_Tp>::type _Up;
    return static_cast<_UP&&>(__t);
}
```
可以看出实际上`std::move`就是去掉参数的引用（如果有的话），然后将其转为右值，因此叫`rvalue_cast`更加容易理解。

### 2. 移动语义的本质

当被问到C++的移动为什么比拷贝快，以及其内部是如何实现的时候。我想当然的认为移动就是，把对象的地址修改为被移动对象的地址。

这其实是经不住推敲的。移动的意思应该是对象被移动过后就不能用原来的地址访问了，然而直接把被移动对象的地址赋给新对象依然能用就地址访问。

实际上，要想知道移动的本质可以直接看看标准库的实现，例如`vector`的移动构造函数：
```c++
template <class _Tp, class _Allocator>
inline _LIBCPP_INLINE_VISIBILITY
vector<_Tp, _Allocator>::vector(vector&& __x)
    : __base(_VSTD::move(__x.__alloc()))
{
    this->__begin_ = __x.__begin_;
    this->__end_ = __x.__end_;
    this->__end_cap() = __x.__end_cap();
    __x.__begin_ = __x.__end_ = __x.__end_cap() = nullptr;
}
```

`std::vector`的移动赋值操作符：
```c++
template <class _Tp, class _Allocator>
inline 
vector<_Tp, _Allocator>&
vector<_Tp, _Allocator>::operator=(vector&& __x)
{
    __move_assign(__x, integral_constant<bool,
          __alloc_traits::propagate_on_container_move_assignment::value>());
    return *this;
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__move_assign(vector& __c, false_type)
{
    if (__base::__alloc() != __c.__alloc())
    {
        typedef move_iterator<iterator> _Ip;
        assign(_Ip(__c.begin()), _Ip(__c.end()));
    }
    else
        __move_assign(__c, true_type());
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__move_assign(vector& __c, true_type)
{
    __vdeallocate();
    __base::__move_assign_alloc(__c); // this can throw
    this->__begin_ = __c.__begin_;
    this->__end_ = __c.__end_;
    this->__end_cap() = __c.__end_cap();
    __c.__begin_ = __c.__end_ = __c.__end_cap() = nullptr;
}

```
从上面可以看到，C++标准库的移动语义只是把类的对象中数据成员中类指针成员所指的数据进行移动。也就是说新对象和被移动对象的地址肯定是不同的，相同的是对象里面的数据。

也就是说如果对象的数据成员没有指针，实际上拷贝和移动的效果就是一样的。这也是为什么`int`之类的基本类型拷贝和移动是一样的。

从上面可以看出，C++标准库的移动语义类似于浅拷贝（只拷贝对象成员数据，不拷贝指针成员所指数据），而赋值语义就是深拷贝（同时拷贝对象指针成员所指数据）。

### 3. 测试



```c++
#include <string>
#include <vector>

int main() {
    std::cout << "======test string===========" << "\n";
    string *s = new string("abcde");
    std::cout << &(*s)[0] << "\n";
    std::cout << &(*s) << "\n";
    string s2 = std::move(*s);
    std::cout << &s2[0] << "\n";
    std::cout << &s2 << "\n";

    std::cout << "=========test vector========" << "\n";
    std::vector<int>* v = new std::vector<int>{1, 2, 3, 4};
    auto itv =  v->begin();
    std::cout << &(*v)[0] << "\n";
    std::cout << &(*v) << "\n";
    std::vector<int> v2  = std::move(*v);
    std::cout << (v2.begin() == itv)  << "\n";
    std::cout << &v2[0] << "\n";
    std::cout << &v2 << "\n";

    std::vector<int> v3(v2);
    std::cout << v3.size() << "\n";
    v3 = std::move(v3);
    std::cout << v3.size() << "\n";

    return 0;
}
```
测试结果：
```
======test string===========
abcde
0x6000037e5120
abcde
0x7ff7be64a688
=========test vector========
0x6000035e0030
0x6000037e5140
1
0x6000035e0030
0x7ff7be64a650
4
0
```