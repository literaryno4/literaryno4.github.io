---
layout: post
title:  "【C++】实现一个简单的反射器"
categories: jekyll update
---

- [引言](#引言)
- [反射实现](#反射实现)
  - [Boost.PP](#boostpp)
  - [元编程](#元编程)
  - [实现](#实现)
- [用法](#用法)

### 引言

假设有这样一个结构体：
```cpp
struct signal {
    int a;
    bool b;
    float c;
    int d;
    //
    // ...
    //
    bool eee;
};

signal s;
```
现在需要`s`进行json序列化，可以使用现有的一些序列化工具，例如 [nlohmann::json](https://github.com/nlohmann/json)。可以这样做：
```cpp
nlohmann::json jsn;
jsn["a"] = s.a;
jsn["b"] = s.b;
jsn["c"] = s.c;
// 
// ...
//
jsn["eee"] = s.eee;
```
可以想象，如果`signal`的数据成员太多（例如上百个），这样机械的代码会重复敲上百次，即使使用一些智能提示工具例如[Tabnine](https://www.tabnine.com/)，也要敲上上百次ENTER键。更重要的是这样的代码太繁琐了，如果后面对结构体添加新的数据成员，相应的序列化代码也要进行添加，增加维护成本。

在写这样的代码的时候，很容易想: 有没有一种方便的方法，可以获取类型的成员名字，然后就可以用循环实现序列化了，类似这样：
```cpp
// int n = get_field(signal).numOfDateMember;
// for (int i = 0; i < n; ++i) {
//     jsn[get_field(signal).names[i]] = get_field(signal).dataMember[i];
// }
```

也就是说，能不能写一个`get_field`函数，让它获取一个类型的成员信息，包括成员的名称、值、类型等信息。

这样的想法很直观，理论上也能够做到，因为这就是编译器做的事：分析表达式各个成分的类型信息，生成token和符号表保存这些类型信息。如果编译器使用的这些信息能够保存至运行时，那么我们自然就有办法获取到，然后使用类似上述方便的序列化方法。

事实上，对于`python`、`java`来说，这样的确是可行的。但是对`C++`而言，就很麻烦了。原因在于，为了追求效率，编译器编译时产生的类型信息并不会保留到运行时，而是在编译完成后就舍弃。这也就意味着，作为 `C++`程序员就不能像其他程序员那样方便地坐享其成了，时至今日， `C++`标准仍未支持这一做法。我们需要使用一些技巧才能实现获取类型信息这一特性。

到这里，就可以引出**反射**这一概念了。反射是指计算机程序在运行时可以访问、检测和修改它本身状态或行为的一种能力。这里的本身状态，就是可以是数据本身的类型信息，包括名称、值和类型。

接下来，尝试实现一个简单的反射，让其支持获取数据的名称和值。

### 反射实现

#### Boost.PP

既然`C++`编译器不保存类型信息至运行时，那么有没有其他办法在编译时或者编译前把我们需要的类型信息保存下来呢。我们知道，`C++`在编译前会进行编译预处理，编译预处理器会对定义的[宏](https://gcc.gnu.org/onlinedocs/cpp/Macros.html#Macros)进行宏展开，生成一系列代码。我们是否可以利用宏来生成有关数据类型信息的代码呢？

答案是可以的，在使用宏参数时，如果前面加一个`#` 符号，编译预处理器就可以把将它替换为该参数的字面名字，比如：
```cpp
#define str(s) #s  // stringizing

signal s;
// ...

std::cout << str(s.a) // 输出 "s.a"
```

至此，我们仿佛找到了方向。因为至少获取一个数据的名称是能办到的。但是，注意到上述代码获取的事"s.a"字符串，但我们只要`s`的成员"a"，这也很容易用编译预处理去处理，后面代码会讲到。

我们可以用宏定义许多有用的宏函数，利用宏展开特性针对我们想要反射的数据生成相应的代码从而获取到数据的类型信息。

为了使代码简单，也可以像使用普通库函数那样调用已有的宏函数。[Boost Preprocessing library](https://www.boost.org/doc/libs/1_78_0/libs/preprocessor/doc/index.html)（Boost.PP）定义了许多方便好用的宏函数，接下来的实现会对用到的函数进行说明。

#### 元编程

我在上一篇讲[模板元编程](https://literaryno4.github.io/cpp-meta-programming.html/)的文章中也说到了代码生成。其实，宏和模板一样都能生成代码均属于元编程范畴。接下来的实现可以看到，除了宏，我们也会用到模板元编程，利用模板的偏特化，可以方便地递归生成代码。

#### 实现

首先自定义一些宏函数，用到了[可变参数](https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html#Variadic-Macros)：
```cpp

#define REM(...) __VA_ARGS__
#define EAT(...)

// Retrieve the type
#define TYPEOF(x) DETAIL_TYPEOF(DETAIL_TYPEOF_PROBE x, )
#define DETAIL_TYPEOF(...) DETAIL_TYPEOF_HEAD(__VA_ARGS__)
#define DETAIL_TYPEOF_HEAD(x, ...) REM x
#define DETAIL_TYPEOF_PROBE(...) (__VA_ARGS__),
// Strip off the type
#define STRIP(x) EAT x
// Show the type without parenthesis
#define PAIR(x) REM x
```
利用模板偏特化，定义类型函数用来给类型添加`const`：
```cpp
// A helper metafunction for adding const to a type
template <class M, class T>
struct make_const {
    typedef T type;
};

template <class M, class T>
struct make_const<const M, T> {
    typedef typename std::add_const<T>::type type;
};
```
接下来使用Boost.PP的宏函数和模板定义一个用于反射的可变参数宏函数：
```cpp
#define REFLECTABLE(...)                                             \
    static const int fields_n = BOOST_PP_VARIADIC_SIZE(__VA_ARGS__); \
    friend struct reflector;                                         \
    template <int N, class Self>                                     \
    struct field_data {};                                            \
    BOOST_PP_SEQ_FOR_EACH_I(REFLECT_EACH, data,                      \
                            BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REFLECT_EACH(r, data, i, x)                                       \
    PAIR(x);                                                              \
    template <class Self>                                                 \
    struct field_data<i, Self> {                                          \
        Self &self;                                                       \
        field_data(Self &self) : self(self) {}                            \
                                                                          \
        typename make_const<Self, TYPEOF(x)>::type &get() {               \
            return self.STRIP(x);                                         \
        }                                                                 \
        typename std::add_const<TYPEOF(x)>::type &get() const {           \
            return self.STRIP(x);                                         \
        }                                                                 \
        const char *name() const { return BOOST_PP_STRINGIZE(STRIP(x)); } \
    };
```
其中`BOOST_PP_SEQ_FOR_EACH_I`宏函数用于序列操作，以第一个参数为函数名生成多段相似（可变参数的数目）代码，其效果可表示为：
```cpp
BOOST_PP_SEQ_FOR_EACH_PRODUCT(h, x, t) // 生成：g(r, x, 0, t0) g(r, x, 1, t1)... g(r, x, k, tk)
```
也就是根据`REFLECT_EACH`展开生成多段类似函数体。

接下来对`reflector`进行定义，我们之后就是用它来获取进行反射获取数据类型信息：
```cpp
struct reflector {
    // Get field_data at index N
    template <int N, class T>
    static typename T::template field_data<N, T> get_field_data(T &x) {
        return typename T::template field_data<N, T>(x);
    }

    // Get the number of fields
    template <class T>
    struct fields {
        static const int n = T::fields_n;
    };
};
```

至此，一个简单的反射器就写好了，接下来讲一下这样一个反射器的用法。

### 用法


