---
layout: post
title:  "Git Tutorial"
categories: jekyll update
---
In `Linux` comand line env, to use git you need to initial a folder:

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
For more info, refer to this [video](https://www.youtube.com/watch?v=SWYqp7iY_Tc)
