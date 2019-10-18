---
layout: post
title:  "Git Tutorial"
categories: jekyll update
---
In `Linux` comand line env, to use [git](https://en.wikipedia.org/wiki/Git) you need to initial a folder:

`~$ git init FOLDER-NAME`

or in the target folder, you just type:

`~$ git init`

---
Then, add files to local git:

`~$ git add FILE-NAME`

if you want to add all:

`~$ git add .`

---
Commit your change:

`~$ git commit -m 'COMMIT-INFO'`

---
Now, you can push your code to github:

`
git remote add origin https://github.com/GITHUB-USER-NAME/REPOSITORY-NAME.git
`

`~$ git push -u origin master`

after that, every time you want to pull, just type:

`~$ git pull`

---
For more info, refer to:
- [video](https://www.youtube.com/watch?v=SWYqp7iY_Tc)
- [Git Cheat Sheet](https://gitee.com/liaoxuefeng/learn-java/raw/master/teach/git-cheatsheet.pdf)
- [Git Official Site](https://git-scm.com/)
- [史上最浅显易懂的Git教程！](https://www.liaoxuefeng.com/wiki/896043488029600)
