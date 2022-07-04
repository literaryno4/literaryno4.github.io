---
layout: post
title:  "好玩的数据结构和算法"
categories: jekyll update
---
### 快速乘法

计算机喜欢做加法而不喜欢乘法，所以可以想办法把乘法转化为加法来提升计算效率。快速乘法本质上就是利用二进制表示和乘法结合率。例如：
$$ 3 \times 5 \\ = 11_2 \times 101_2\\ = 11_2 \times (100_2 + 10_2 + 1_2)\\ = 11_2 \times 100_2 + 11_2 \times 10_2 + 11_2 \times 1_2 \\ = 11_2 << 2 + 11_2 << 1 + 11_2 $$

算法实现：
```c++
long long quickMult(long long a, long long b) {
        long long res = 0;
        while (b > 0) {
            if (b & 1) {
                res += a;
            }
            b >>= 1;
            a <<= 1;
        }
        return res;
}
```

类似的还有快速幂运算。

### 树状数组（Binary Indexed Tree）

有这样一个需求：

- 保存n个数据，每个数据都可能更新；
- 随时查询这个数组第1～i个元素的和

最简单的方式是直接用一个数组，那么更新的复杂读为$O(1)$，查询的复杂度为$O(n)$。使用树状数组可以将查询的复杂读将为$O(\log{n})$，更新的复杂度也为$O(\log{n})$。

简单来说，树状数组利用数的二进制表示来保证查询复杂读为$O(\log_{2}{n})$，例如$\sum_{1}^{7} = tree[7] + tree[6] + tree[4]$，我画了一个简单示意图，其中方块表示第1～i个元素的和：
![](/assets/bit.jpg)

网络上的一张示意图：
![](/assets/bit2.png)

也就是说，tree[i]表示i之前的$2^k$元素之和，其中k是i的二进制表示的右边第一个1之后的零的个数：
$$ 2^k = i \And (\sim i) $$

一个树状数组的实现如下：
```c++
class BIT {
    vector<int> tree_;
    int n_;

    public:
    BIT(int n) : n_(n), tree_(n_ + 1) {}
    void update (int i) {
        while (i <= n_) {
            tree[i]++;
            i += lowbit(i);
        }
    }

    int query(int i) {
        int res = 0;
        while (i > 0) {
            res += tree[i];
            i -= lowbit(i);
        }
        return res;
    }

    private:
    int lowbit(x) {
        return x & (-x);
    }
}
```
LeetCode上，树状数组可以用来[计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/)、[数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)。
