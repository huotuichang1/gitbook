---
description: Beamforming
---

# 第四章

        Beamforming,波束形成。前面我们介绍了如何对声源位置进行确认和追踪，但我们应该通过什么方式对我们所关注的目标方向的声音信息进行增强呢？答案就是波束形成技术。无论是什么形式的波束形成，其效果总是通过几个麦克风的信号组合来加强目标方向声源信息和抑制来自其他方向的干扰信号。那么这一效果是通过什么方式方法进行实现的呢？下面将为大家一一介绍。

## 1.波束形成基本原理与概念

### 1.1声音的传播和阵列几何位置

        我们首先来思考一个具有 $$N$$ 个传感器的随机形状的阵列。 我们假设这 $$N$$ 个传感器的坐标 $$\vec m_n$$ 全部已知， $$n=0,1,...,N-1$$ 。传感器接收到的信号可以表示如下：

$$
\vec f(t,\vec m)=\left[\begin{aligned}f(t,\vec m_0)\\
f(t,\vec m_1)\\
.\quad\quad\\
.\quad\quad\\
f(t,\vec m_{N-1})\end{aligned}\right]
$$

        现在要注意到我们假定是在连续时域 $$t$$ 上进行处理。这是为了避免由于离散时间而导致的间隔。但是这个问题在我们将宽带波束在分割为窄带变为不同子带域的时候会消失。因为我们在子带域中应用的相移和比例因子是连续的值，不管开始的信号是否是这样。每个麦克风的输出是通过一个线性时不变滤波器和冲击响应 $$h_n(\tau)$$ 进行滤波求和输出的。

$$
y(t)=\sum_{n=0}^{N-1}\int_{-\infty}^{\infty}h_n(t-\tau)f_n(\tau,\vec m_n)d\tau
$$

        在矩阵形式中，延迟求和滤波器传感器\(麦克风\)权重可被表示如下：

$$
y(t)=\int_{-\infty}^{\infty}\vec h^T(t-\tau)\vec f(\tau,\vec m)d\tau
$$

        其中

$$
\vec h(t)=\left[\begin{aligned}h_0(t)\\
h_1(t)\\
{.}{\quad}\\
{.}{\quad}\\
h_{N-1}(t)\end{aligned}\right]
$$

        使用连续时间傅里叶变换转移到频域上可以重写为：

$$
Y(\omega)=\int^{\infty}_{-\infty}y(t)e^{-j\omega t}dt=\vec H^T(\omega)\vec F(\omega,\vec m)
$$

        注意这里的分解 $$Y(\omega)=\vec H^T(\omega)\vec F(\omega,\vec m)$$ 类似于时域上的形式，其分别可以写为

$$
\vec H(\omega)=\int^{\infty}_{-\infty}\vec h(t)e^{-j\omega t}dt\\
\vec F(\omega,\vec m)=\int^{\infty}_{-\infty}\vec f(t,\vec m)e^{-j \omega t}dt
$$



