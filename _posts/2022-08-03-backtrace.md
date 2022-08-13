---
layout: post
title:  "【操作系统】GDB的backtrace原理"
categories: jekyll update
---

<!-- vim-markdown-toc GFM -->

* [栈帧](#栈帧)
* [backtrace](#backtrace)
* [简单的实验](#简单的实验)

<!-- vim-markdown-toc -->

### 栈帧

当用户调用函数时就会产生一个栈帧，包含返回地址，寄存器值和局部变量等信息。函数退出是则会删除此栈帧。这本质上就是对栈指针（stack pointer） 寄存器的修改。例如一个用户栈是从高位地址向低位地址增长，那么函数的代码（汇编角度看）运行前就会减小栈指针的值（例如16，这取决函数使用的变量的多少），栈指针增长的这段空间就是一个栈帧。相反，当运行完函数时会增大栈指针的值（与上述减小的值相对应），这就意味着删除来这一函数的栈帧。

如果一个函数内调用另外函数或者递归，那么会在现有栈帧之上生成新的栈帧，直到最上层的函数返回，反向删除已有栈帧。

### backtrace

今天主要是想讲一讲我发现的GDB调试时backtrace指令的原理，其他调试器的原理应该也是如此，因为它是基于大多数处理器都提供的一个寄存器实现的。我们都很熟悉栈指针，但是常常忽略另外一个帧指针（frame pointer）。它看起来好像特别没有存在感，我之前甚至都没注意。实际上它和栈指针一样，无处不在，它保存当前栈帧的位置，即栈帧顶部位置。当调用一个函数的时候，我们不仅会在栈上保存返回地址，还会保存前一个栈帧地址（前一个栈的栈顶位置，也即是前一个帧指针）。这看起来好像多此一举没什么用，因为我们想找到前面的栈，只需要弹栈即可。但是，如果程序在当前函数崩溃了呢，我们想要追溯当前的所用栈帧该如何做，这就是帧指针的作用。

我们可以看下图所示的栈帧，可以看到，实际上，通过帧指针，我们形成了从低位上层函数栈帧到高位底层函数栈帧的链表，每个栈帧节点顶部偏移（-16）处就是指向前一个栈帧的指针。这样我们就能轻松追溯任意位置的所有调用栈帧了。这就是一般调试器的backtrace功能的原理了。

### 简单的实验

我们用GDB简单的看一看backtrace的作用：
```c 
int a() {
  int x = 0, y = 1;
  int* p = NULL;
  return *p; // segment fault
}

void b() {
  int x = 1, y = 2;
  a();
}


int main() {
  b();
  return 0;
}
```

使用GDB运行程序到发生段错误，然后运行backtrace命令，得到结果如下：
```
(gdb) backtrace
#0  0x0000555555555188 in a () at backtrace.c:7
#1  0x00005555555551c9 in b () at backtrace.c:13
#2  0x00005555555551de in main () at backtrace.c:18
```
可以看到，地址从main函数（高）到a函数（低）成功backtrace整个调用栈，再看看函数a的栈帧：
```
(gdb) info frame
Stack level 0, frame at 0x7fffffffe370:
 rip = 0x555555555188 in a (backtrace.c:7); saved rip = 0x5555555551c9
 called by frame at 0x7fffffffe390
 source language c.
 Arglist at 0x7fffffffe348, args:
 Locals at 0x7fffffffe348, Previous frame's sp is 0x7fffffffe370
 Saved registers:
  rbp at 0x7fffffffe360, rip at 0x7fffffffe368
```
可以发现的确保存了前一个函数（b）的栈帧位置（Previous frame's sp）进一步发现函数的代码地址（`0x555555555188`）明显低于函数栈的位置（`0x7fffffffe370`），这也很符合预期，详细可看之前的[系统调用文章]()。




