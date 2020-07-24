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
