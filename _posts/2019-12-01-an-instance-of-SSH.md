---
layout: post
title:  "An Instance of SSH"
categories: jekyll update
---

Reccently, I find a chance to rent servers from [GCP](https://console.cloud.google.com). And I try to connect my server using SSH locally. Through I had used SSH before when I using git to push my code to github, I do not understand how it works. Here is now what I am trying to make sense.

## How I connect my server

- generate public key and private key in **~/.ssh/**:

```Shell
~$ ssh-keygen -t rsa -f ~/.ssh/gcloud -C chao
```

> where **gcloud** is the name of generated keys and chao is the user name to log in.

- Copy the content of **gcloud.pub** file to the GCP server.

- Then, we can use local terminal to log in GCP server:

```Shell
~$ ssh -v -i ~/.ssh/gcloud chao@IPOFSERVER
```

> where **-v** is used to show debug message and **-i** is used to select a identity file. **chao** is the name of user.

## What is SSH(Secure Shell)

[this blog](https://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html) and [this page](https://en.wikipedia.org/wiki/Secure_Shell) say that SSH is a kind of cryptographic network protocol for cryptographically logon between computers.

We can find SSH uses [public-key cryptography](https://en.wikipedia.org/wiki/Public-key_cryptography) like [https](https://en.wikipedia.org/wiki/HTTPS).
