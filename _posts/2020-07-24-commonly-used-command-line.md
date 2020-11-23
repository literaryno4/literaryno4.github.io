---
layout: post
title:  "常用Linux命令（持续更新）"
categories: jekyll update
---

- 多任务管理: `fg`, `bg`, `jobs`, `&`;

- 远程连接执行长时间任务，防止意外断开：`screen`;
  - 建立并进入一个screen:
  ```shell
  screen -S NAME
  ```
  - 保留当前screen并跳出: `<Ctrl + a> d`

  - 列出所有保留的screen（包含对应ID）:
    ```shell
    screen -ls
    ```
  - 返回到保留的一个screen中:
    ```shell
    screen -r ID
    ```

- 多文件重命名: `rename`;
  - 将文件后缀`A`改为`B`:
  ```shell
  rename 's/.A/.B/'
  ```
  
- 获取当前文件夹下文件数目：
  - 获取文件数目:
    ```shell
    ls -l | grep "^-" | wc -l
    ```
  - 获取文件数目(包括文件夹）：
    ```shell
    ls -lR | grep "^-" | wc -l
    ```
  - 获取文件夹数目：
    ```shell
    ls -lR | grep "^d" | wc -l
    ```
 
- 用户管理: `useradd`, `groupadd`, `su`;
  - 创建用户:`useradd`:
    ```shell
    useradd -m USERNAME
    ```
    or
    ```shell
    adduser USERNAME
    ```
  - 删除用户:
    ```shell
    useradd -r USERNAME
    ```
  - 切换用户: `su`;
    ```shell
    sudo su
    su USERNAME
    ```
    
- 列出非指定文件：
  - 单个文件:
    ```shell
    ls -I FILENAME`:
    ```
  - 忽略扩展名`*.png`:
    ```shell
    ls -I "*.png"
    ```
