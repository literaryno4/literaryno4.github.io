---
layout: post
title:  "Neovim and Lua"
categories: jekyll update
---
### Vim + YouCompleteMe
这两天一直在搞vim编辑器，起因其实只是我听别人又提起了YouCompleteMe(YCM)，那个曾经的噩梦（主要还是网络问题，国内实在是不方便，有些依赖下不下来）。但是它还是勾起了我的好奇心，而且我的低配处理器17年MacBook Pro夏天跑CLion确实有些吃力了。忍不住想尝试一下，不试不知道，还是以前的网络问题，我又花了一天时间才配好，真浪费时间，之间等待下载、编译、安装的时间又无法安下心。实在是得不偿失。好在现在可以愉快地使用了。末尾我放一个现在的vim配置文件。

### Neovim

然而，故事到这儿才刚刚开始。我安装YCM好了才发现，现在大家都不太玩YCM了，好多人都用[coc.nvim]()了，甚至不玩Vim 8.0了，而是用一个兼容于Vim 8.0的Neorvim(nvim)，它最大的好处在于：

- 兼容Vim配置的同时，还可以使用lua进行配置。lua的简单灵活使nvim短时间内就爆火，社区活跃度都超过Vim了；
- 原生支持LSP。如今虽然nvim版本才到0.8，但是已经有大批优秀的插件了，其中最重要的是nvim开始原生支持Language Server Protocol（LSP），这是微软的一个用于代码编辑（包括静态检查、代码补全）的服务器协议，这意味着可以像IDE一样实现流畅跟手的代码提示功能，且如果本地安装也不用担心网络问题；
- 支持Tree-sitter。这是一个优秀的parser，可以代码编辑时实时高效地更新生成语法树，支持多种编程语言、无依赖。这也以为这，它是一款优秀的语法高亮，代码折叠插件，用过之后不得不说是真的好。似乎Vim还不支持Tree-sitter。

于是乎，我又花了两天的时间倒腾Neovim，时间主要花在学习lua语言和配置各种插件。nvim主要是在我的腾讯云Ubuntu服务器配置，仍然会有恶心人的网络问题，因为好多插件没有国内源，实在太慢了，好在最终配好了。配置主要是参考[Neovim-from-scratch](https://github.com/LunarVim/Neovim-from-scratch)，搞懂了配置原理后删除了不需要的部分，修改添加了子集需要的，末尾也放一个lua配置文件链接。nvim配置是全lua文件的，虽然可以和vim一样用vimscript，但是lua是真的简单易读，风格统一，而且感觉lua是nvim的大杀器，大势所趋，不玩lua还不如用vim+YCM。虽然对lua就是了解的水平，但这次不正是学习的大好时机。所以接下来我重点总结整理下lua相关知识。

### Lua：强大、高效、轻量、可嵌入脚本语言

第一次听说Lua是云风的博客上看到的，了解到其可以和C语言结合使用，游戏开发用得很多，然后Nginx也是用Lua做配置。选择Lua的原因主要有：语法简单，功能齐全，以及提供好用的C API。

#### 简单的语法

1. chunks

若干语句构成的lua代码就是一个chunk，例如：
```lua
-- chunk 1
print("hello, world\n")
```
```lua
-- chunk 2
a = io.read("*number")
print(a)
```

2. 变量

lua默认全局变量，使用`local`声明局部变量：
```lua
s = "yahaha" -- global variable

local n = 100 -- local variable
```

3. 语句

```lua
-- assignment
a, b, c = 0, 1, 2

-- if
if a < b then
  print (a)
elseif a == b then
  print(b)
else
  print(c)
end

-- while loop
while i <= 10 do
  local x = i*2
  print(x)
  i = i + 1
end

-- for loop, print 10, 9, 8 ... 1
for j = 10, 1, -1 do
  print(j)
end
```

4. 函数

```lua
function funName(arg1, arg2)
  return arg1, arg2
end

-- closures
function newCounter ()
  local i = 0
  return function () 
           i = i + 1
	   return i
	 end
end
```
其他详细的语法细节可以参考[Learn X in Y minutes](https://learnxinyminutes.com/docs/lua/)和[Programming in Lua (first edition)](http://www.lua.org/pil/contents.html)

#### C API

### 参考文献

- [我的nvim配置]()
- [我的vim + YCM配置]()

