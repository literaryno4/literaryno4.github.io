---
layout: post
title:  "Run A Mininal Riscv Linux in Qemu"
categories: jekyll update
---

Table of content:
<!-- vim-markdown-toc GFM -->

* [prerequisites](#prerequisites)
* [root file system](#root-file-system)
* [Compile linux kernel](#compile-linux-kernel)
* [run in qemu](#run-in-qemu)
* [other details](#other-details)
    * [initrd](#initrd)
    * [hello kernel](#hello-kernel)
* [reference](#reference)

<!-- vim-markdown-toc -->

### prerequisites

- [qemu-system-riscv64](https://github.com/qemu/qemu
).

- [riscv-tools](https://github.com/riscv-software-src/riscv-tools).

- [busybox](https://www.busybox.net/about.html).

- [linux source code](https://github.com/torvalds/linux
).

### root file system

To run a minimal riscv linux, we need a rootfs firstly. Here I polulate rootfs with busybox, a tiny version of many Unix utilities, and some other must included directories and files.

Turn on static link option and compile:
```shell
cd busybox
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig

# now open kconfig, select Settings -->
#                   Build static binary (no shared libs)
#                   Press Y
#                   save and exit
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig

# make
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j $(nproc)
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- install
```
Now we have many riscv64 binaries in `_install` which will run on riscv linux and be as part of the root file system.

It look like this:
```shell
tree . -d 1
> .
> ├── bin
> ├── sbin
> └── usr
>     ├── bin
>     └── sbin
```

next, create another directories:
```shell
mkdir dev proc etc lib tmp
```

All these minimum set of directories is suggested by [linux docs](https://tldp.org/HOWTO/Bootdisk-HOWTO/buildroot.html#AEN315).

Note that the `lib` is not necessary because we statically compile the busybox. And `tmp` is also not necessary.

We leave the directories we create empty except for `dev` in which we must create devices to be used by binaries:
```shell
# create new devices with `mknod`
cd dev
sudo mknod console c 5 1 
cd ..

# or cp from your linux machine's dev directories
sudo cp -R /dev/console dev
```

and don't foget a init program to tell kernel where to begin with.

save these script as init:
```shell
#!/bin/sh

mount -t proc none /proc

echo -e "\nBoot took $(cut -d' ' -f1 /proc/uptime) seconds\n"
```
make it be execuable:
```shell
chmod +x init
```

Now we have a minimum root file system to be run by riscv linux.

### Compile linux kernel

The simplest way to run linux in qemu is use initramfs which allocate ram for the above root file system to init kernel.
After linux 2.6, the linux kernel always create a gzipped cpio format initramfs archive and links it into the resulting kernel Image. Note that this archive is empty by default which is why we create it above from busybox.

To use generate the customized initramfs from root file system we create above, we firstly should do some config:

```shell
cd linux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig

# open kconfig and select general setup ---> and find
#                  Initial RAM filesystem and RAM disk(initramfs/initrd) support
#                  press Y 
#                  then select Initramfs source file(s)
#                  press enter
#                  input path to your rootfs(../busybox/_install)
#                  save and exit
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig

# make
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j $(nproc)
```

Now we have the linux Image to boot in `linux/arch/riscv/boot/`.

### run in qemu

Now we can run the compiled riscv linux in qemu:
```shell
qemu-system-riscv64 \
-machine virt \
-nographic \
-kernel arch/riscv/boot/Image
```

A riscv linux run in qemu look like this:
```
...
[    0.440329] usbcore: registered new interface driver usbhid
[    0.440569] usbhid: USB HID core driver
[    0.442874] NET: Registered protocol family 10
[    0.455901] Segment Routing with IPv6
[    0.456785] sit: IPv6, IPv4 and MPLS over IPv4 tunneling driver
[    0.459711] NET: Registered protocol family 17
[    0.461160] 9pnet: Installing 9P2000 support
[    0.461787] Key type dns_resolver registered
[    0.462668] debug_vm_pgtable: [debug_vm_pgtable         ]: Validating architecture page table helpers
[    0.498488] Freeing unused kernel memory: 3292K
[    0.500666] Run /init as init process

Boot took 0.57 seconds

/bin/sh: can't access tty; job control turned off
~ #
~ #
~ # ls
bin      etc      lib      linuxrc  proc     sbin     usr
dev      init     lib64    mnt      root     sys
~ #
```

### other details

#### initrd

In fact, you can leave default config and compile linux kernel and specify initramfs using qemu option like this:
```shell
qemu-system-riscv64 \
-machine virt \
-nographic \
-kernel arch/riscv/boot/Image
-initrd /path/to/initramfs.cpio
```

Here the `initramfs.cpio` can be created from the root file system above like this:
```shell
cd ../busybox/_install
find . | cpio -H newc -o > ../../initramfs.cpio
```

with `initrd`, the initramfs defaultly linded in kernel Image will be override.

#### hello kernel

We can make kernel run any program other than `/bin/sh`.
```shell
cat > hello.c << EOF
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
  printf("Hello world!\n");
  sleep(999999999);
}
EOF
riscv64-linux-gnu-gcc -static hello.c -o init
echo init | cpio -o -H newc | gzip > test.cpio.gz 
# Testing external initramfs using the initrd loading mechanism.
qemu -kernel /boot/vmlinuz -initrd test.cpio.gz /dev/zero
```

Here the kernel find the hello binary named init and execute it to print `Hello World!`. This small root file system with only one init binary is truely a minimal riscv linux!.

### reference

- [Ramfs, rootfs and initramfs](https://www.kernel.org/doc/html/latest/filesystems/ramfs-rootfs-initramfs.html)

- [Building a root filesystem](https://tldp.org/HOWTO/Bootdisk-HOWTO/buildroot.html)

- [‘virt’ Generic Virtual Platform](https://www.qemu.org/docs/master/system/riscv/virt.html)


