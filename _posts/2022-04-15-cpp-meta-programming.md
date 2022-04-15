---
layout: post
title:  "【C++】元编程初探"
categories: jekyll update
---

- [引言](#引言)
- [元编程的选择控制结构](#元编程的选择控制结构)
  - [Conditional 二选一](#conditional-二选一)
  - [Select 多选一](#select-多选一)
- [元编程的迭代和递归](#元编程的迭代和递归)
- [总结](#总结)
- [参考文献](#参考文献)

### 引言

考虑这样的一个需求，有一个函数fun针对类型是否有拷贝构造函数可以有不同的实现方式，假如A有拷贝构造函数, B无拷贝构造函数， 有拷贝构造函数的实现性能更为高效：
```c++
class A {
public:
    A() : i(1) {}
    int i;
};

class B {
public:
    B() : i(1) {}
    B(const B&) = delete;
    int i;
};

template<typename T>
void fun(T& t) {
    std::cout << "fun for copy constructible\n";
    std::cout << "below is optimized implementation\n";

    // ...

}

template<typename T>
void fun(T& t) {
    std::cout << "fun for copy non-constructible\n";
    std::cout << "below is common implementation\n";

    // ...

}
```

如何让编译器知道根据是否有拷贝构造函数实例化相应的fun实现呢？

C++标准库提供的`enable_if`元函数（meta function）它能够根据某些条件是否满足（例如是否有拷贝构造）来决定声明并定义哪一个符号，统称条件定义。

`enable_if`的实现相当简单：
```c++
template<bool B, typename T = void>
struct enableIf {
    typedef T type;
};

// 偏特化，B为false的时候没有type类型成员
template<typename T>
struct enableIf<false, T> {};
```

如果`eableIf`的第一个参数为`false`就偏特化，此时没有`type`类型成员，利用这个特性，就可以实现条件定义：
```c++
template<typename T>
typename enableIf<std::is_copy_constructible<T>::value>::type fun(T& t) {
    std::cout << "fun for copy constructible\n";
    std::cout << "below is optimized implementation\n";

    // ...

}

template<typename T>
typename enableIf<!std::is_copy_constructible<T>::value>::type fun(T& t) {
    std::cout << "fun for copy non-constructible\n";
    std::cout << "below is common implementation\n";

    // ...

}
```
`std::is_copy_constructible<T>`也是一个元函数，顾名思义，它的value就是指明`T`类型是否有拷贝构造函数。上述实现中，如果`enableIf`第一个模板参数为false，有type类型成员，例如：
```c++
int main() {
    A a;
    fun(a);

    return 0;
}
```
因为 A 有拷贝构造函数，所以`enableIf`有type类型成员，编译器定义优化版本的fun函数。输出：
```
fun for copy constructible
below is optimized implementation
```


如果`enableIf`第一个模板参数为false，就没有type类型成员，例如：
```c++
int main() {
    B b;
    fun(b);

    return 0;
}
```
因为`B`没有有拷贝构造函数，所以`enableIf`没有有type类型成员，编译器定义普通fun函数。输出：
```
fun for copy non-constructible
below is common implementation
```

实际上，可以将`typename enableIf<std::is_copy_constructible<T>::value>::type`看成是一个类型函数`EnableIf：
```c++
template<bool B, typename T = void>
using EnableIf=typename enableIf<B, T>::type;
```
它的参数列表在`<>`中，可以是值或者类型，输出是一个类型。这样上述的fun函数可写为更直观的版本：
```c++
template<typename T>
EnableIf<std::is_copy_constructible<T>::value> fun(T& t) {
    std::cout << "fun for copy constructible\n";
    std::cout << "below is optimized implementation\n";

    // ...

}

template<typename T>
EnableIf<!std::is_copy_constructible<T>::value> fun(T& t) {
    std::cout << "fun for copy non-constructible\n";
    std::cout << "below is common implementation\n";

    // ...

}
```

到这里，你应该对元编程有一点直觉的认识了。利用模板编写一个类型函数，将它作为生成器，在编译时生成类型和函数，这就是元编程了。相比而言，泛型编程也使用模板，但它更强调的是编写通用的类或者函数，而元编程强调编译时计算（比如类型函数的计算）。

实际上，利用模板和实例化机制进行元编程可以做到任何其他语言能做的事。接下来你将看到元编程有控制结构，可以完成编译时迭代（以递归的形式），它是图灵完备的，可以把它看成是一门编译时函数式编程语言。

### 元编程的选择控制结构

#### Conditional 二选一
如果要在两个类型中进行选择，可以实现一个`conditional`类型函数，它就像`?:`运算符对两个值进行选择一样对两个类型进行选择。`conditional`实现也很简单：
```c++
template<bool C, typename T, typename F>
struct conditional {
    using type = T;
};

template<typename T, typename F>
struct conditional<false, T, F> {
    using type = F;
};

template<bool C, typename T, typename F>
using Conditional = typename conditional<C, T, F>::type;
```
例如，想根据类型`Ty`是否是多态类型，返回相应的类型`X`，`Y`来定义一个对象`z`，就可以这样做：
```c++
Conditional<(Is_polymorphic<Ty>()), X, Y> z;
```
如果为是多态，就：
```c++
X z;
```
不是多态，就：
```c++
Y z;
```

#### Select 多选一

类似Conditional，Select类型函数根据第一个非类型参数`N`返回第N个类型：
```c++
template<unsigned N, typename ... Cases>
struct select;

template<unsiged N, typename T, typename ... Cases>
struct select<N, T, Cases...> : select<N-1, Cases...> {
};

template<typename T, typename ... Cases>
struct select<0, T, Cases...> {
    using type = T;
}

template<unsigned N, typename ... Cases>
using Select = typename select<N, Cases...>::type;
```
上述实现用到了可变参数模板，和普通递归一样，有一个下界条件，当到达`N = 0`时，使用偏特化版本`select`返回`type`。

`Select`类型函数一个实际用处是配合元祖[`std::tuple`](https://en.cppreference.com/w/cpp/utility/tuple)使用。

###  元编程的迭代和递归

在编译时不能使用变量，所以元编程一般使用递归实现编译时迭代。例如一个阶乘函数模板：
```c++
template<int N>
constexpr int fac() {
    return N * fac<N - 1>();
}

template<>
constexpr int fac<1>() {
    return 1;
}

constexpr int x5 = fac<5>();
```
可以看到，和函数式编程一样，元编程处理一系列值的方式是递归调用，知道终止条件。这里的终止条件就是一个特例化函数模板。

### 总结

可以看到，元编程可以在编译时做任何计算。但是应该注意，它具有代码易读性差、调试难读高等缺点。所以在使用元编程前要考虑把计算提前到编译时是否值得。如下场景是值得的：

- 做安全检查。如[Effective C++](https://book.douban.com/subject/5387403/)而言，把程序的错误尽量提前永远是值得的；
- 提高类型安全。计算数据的确切类型，消除很多显示类型转换；
- 提高运行时性能。在编译时选择运行时要调用的函数，或者计算运行时需要的数据。

### 参考文献

- [C++程序设计语言](https://book.douban.com/subject/26857943/)