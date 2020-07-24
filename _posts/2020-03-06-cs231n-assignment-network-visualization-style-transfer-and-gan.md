---
layout: post
title:  "CS231n Assignments Note9: Network Visualization, Style Transfer and Generative Adversarial Networks"

categories: jekyll update
---

Table of Contents:

- [A Try to Interprete The Essence of RNN](#inter)

- [Use LSTM to Improve RNN](#lstm)

- [Generative Adversarial Networks](#gan)

- [Reference](#rf)

<a name='gan'></a>

## Generative Adversarial Networks

**discriminator**: a traditional classification network;

**generator**: take random noise as input and transform it to produce images to fool the discriminator into thinking the images it produced are real.

We can think this back and force process of the generator($$G$$) trying to fool the discriminator($D$), and the discriminator tring to correctly classify real vs. fake as a minimax game:

$$
\underset{G}{\operatorname{minimize}} \underset{D}{\operatorname{maximize}} \mathbb{E}_{x \sim p_{\text {data }}}[\log D(x)]+\mathbb{E}_{z \sim p(z)}[\log (1-D(G(z)))]
$$

From [Goodfellow et al.](https://arxiv.org/abs/1406.2661), we alternate the following updates:

- update the generator to maximize the probability of the discriminator making the incorrect choice on generated data:

$$
\underset{G}{\operatorname{maximize}} \mathbb{E}_{z \sim p(z)}[\log D(G(z))]
$$

- update the discriminator to maximize the probability of the discriminator making the correct choice on real and generated data:

$$
\underset{D}{\operatorname{maximize}} \mathbb{E}_{x \sim p_{\text {data }}}[\log D(x)]+\mathbb{E}_{z \sim p(z)}[\log (1-D(G(z)))]
$$

<a name='rf'></a>

## Reference

[CS231n](http://cs231n.stanford.edu/)

[literaryno4/cs231n](https://github.com/literaryno4/cs231n)

[min-char-rnn.py gist: 112 lines of Python](https://gist.github.com/karpathy/d4dee566867f8291f086)

[Lecture 10: Recurrent Neural Networks](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture10.pdf)(slide)

[A Useful Video](https://www.bilibili.com/video/av86713932?p=17)

