---
layout: post
title: "Create My VPS"
categories: jekyll update
---

I follow this [video](https://www.youtube.com/watch?v=RxbGtkRVUWQ&t=377s) and [page](https://zhuanlan.zhihu.com/p/55893138) to create my VPS(Virtual Private Server) for SSR and testing web server. I'd like to note some details:

- Using `nmap` for port scanning and security auditing.

```Shell
# to scan local port
~$ nmap 127.0.0.1
```
- Install and start a apache2 server:

```Shell
~$ sudo apt-get install apache2
~$ sudo /etc/init.d/apache2 start
```

