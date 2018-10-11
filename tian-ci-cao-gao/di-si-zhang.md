---
description: Beamforming
---

# 第四章

        Beamforming,波束形成。前面我们介绍了如何对声源位置进行确认和追踪，但我们应该通过什么方式对我们所关注的目标方向的声音信息进行增强呢？答案就是波 束形成技术。无论是什么形式的波束形成，其效果总是通过几个麦克风的信号组合来加强目标方向声源信息和抑制来自其他方向的干扰信号。那么这一效果是通过什么方式方法进行实现的呢？下面将为大家一一介绍。

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

        他们分别是滤波器的频率响应矢量和麦克风信号的频谱。

        \(下面这里看起来比较复杂，其实是将宽带信号分割为窄带再还原回去的过程\)在构建一个实际的波束形成系统时，我们是不能使用上面使用的连续时间傅里叶变换的。相对应的，是对每个麦克风的输出进行采样然后进入（傅里叶）分析滤波器组进行处理，产生一组子带。这些样本是中心频率为 $$\omega_m=2\pi m/M$$ 的 $$N$$ 个样本，其中 $$M$$ 是子带的数目。然后依次相加然后求内积。！这个地方可能有疑问，如下：

![](../.gitbook/assets/20181011-192816-ping-mu-jie-tu.png)

低通滤波器的原型滤波器做为滤波器组，另外 $$M-1$$ 个滤波器原型滤波器的调制，调制因子是 $$e^{-j\omega_m n},\omega_m=2\pi m/M,m=1,2,3...,M$$ 对应刚刚的中心频率。然后所有 $$M$$ 个波束形成器的输出可以由合成滤波器组转化回时域。其中分析滤波器组只有在满足信号的采样足够频繁满足奈奎斯特准则的时候才可以看做是采样信号的短时傅里叶变换，（详细过程见滤波器组章节描述）子带频域上的波束形成有很大的优点，主动传感器权重可以独立地针对每个子带优化。这样处理的好处是相对于相同长度的时域滤波求和波束形成节省了很大的运算量。一个简单的滤波器组的了解可以参考这两篇文章：1.（[https://blog.csdn.net/shichaog/article/details/77379998](https://blog.csdn.net/shichaog/article/details/77379998)）。2.（[https://blog.csdn.net/book\_bbyuan/article/details/80366196](https://blog.csdn.net/book_bbyuan/article/details/80366196)），详细了解建议参考《distantspeechrecognition》中的第十一章。

        另外值得一提的是虽然

