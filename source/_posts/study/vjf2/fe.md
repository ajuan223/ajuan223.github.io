---
title: 期末考知识点查漏
date: 2025-06-06 15:00:00
---
## 级数
1. 收敛半径： 
    - 比值 $r=\lim_{n\to\infty} \frac{|a_n|}{|a_{n+1}|}$
    - 根值 $r=\lim_{n\to\infty} n\sqrt{|a_n|}$
---
2. 幂级数展开
- **指数函数**
$$e^x=\sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{1}{2!}x^2 + \frac{1}{3!}x^3 + \cdots (R=+∞)$$
   其中 $R$ 为收敛半径。
- **三角函数**
$$sinx=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{(2n+1)!} x^{2n+1} = x - \frac{1}{3!}x^3 + \frac{1}{5!}x^5 - \cdots (R=+∞)$$
$$cosx=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{(2n)!} x^{2n} = 1 - \frac{1}{2!}x^2 + \frac{1}{4!}x^4 - \cdots (R=+∞)$$
   其中 $R$ 为收敛半径。
$$arctanx=\sum_{n=0}^{\infty} \frac{(-1)^{n}}{2n+1} x^{2n+1} = x - \frac{1}{3}x^3 + \frac{1}{5}x^5 - \cdots (|x|<1)$$
   其中 $R$ 为收敛半径。
- **对数函数**
$$ln(1+x)=\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} x^n = x - \frac{1}{2}x^2 + \frac{1}{3}x^3 - \cdots (|x|<1)$$
   其中 $ln(1+x)$ 的收敛半径为 1。
- **二项式展开**
$$(1+x)^\alpha =\sum_{n=0}^{\infty} \binom{n}{\alpha} x^n =1+\alpha x+\frac{1}{2!}\alpha(\alpha−1)x^2 +⋯(∣x∣<1)$$
其中 $\binom{n}{\alpha}=\frac{n!}{\alpha(\alpha−1)⋯(\alpha−n+1)}$。
---
3. 和函数
$$\sum_{n=1}^{\infty} x^n=\frac{x}{1-x}$$
   其中 $|x|<1$。
---
4. - **傅里叶级数**
$$f(x)=\frac{a_0}{2}+\sum_{n=0}^{\infty} [a_n \cos(nx) + b_n \sin(nx)]$$
其中 $a_n=\frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)dx$，$b_n=\frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)dx$。
*$$f(x)=\sum_{n=-\infty}^{\infty} c_n e^{inx}$$
其中 $c_n=\frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}dx$。*
   - **狄利克雷收敛定理：**
      - 连续或有限个间断点
      - 有限个极值点

## 空间解析几何
1. - **切平面方程**
   $$F(x_0, y_0, z_0) = 0$$
   $$\frac{\partial F}{\partial x}(x-x_0) + \frac{\partial F}{\partial y}(y-y_0) + \frac{\partial F}{\partial z}(z-z_0) = 0$$
   其中 $F(x, y, z)$ 是曲面的方程，$(x_0, y_0, z_0)$ 是切点，$\frac{\partial F}{\partial x}$、$\frac{\partial F}{\partial y}$ 和 $\frac{\partial F}{\partial z}$ 是该点的偏导数。
   对应的**法线方程**为：
   $$\frac{x-x_0}{F'_x} = \frac{y-y_0}{F'_y} = \frac{z-z_0}{F'_z}$$
   - **切向量**
   $$\mathbf{n_1} = \left( \frac{\partial F}{\partial x}, \frac{\partial F}{\partial y}, \frac{\partial F}{\partial z} \right)$$
   $$\mathbf{n_2} = \left( \frac{\partial G}{\partial x}, \frac{\partial G}{\partial y}, \frac{\partial G}{\partial z} \right)$$
   $$\mathbf{n} = \mathbf{n_1} \times \mathbf{n_2}$$
---
2. - **投影柱面**
   设点 $P(x_0, y_0, z_0)$, $\mathbf{PP_0} \parallel \mathbf{n}$, 又有点 $P_0(x_0, y_0, 0)$在给定平面上,联立解出投影柱面方程.
   - **投影曲线**
   投影柱面与平面的交线.


## 多元函数微分学
1. 方向导数
   定义:函数定义域的内点对某一方向求导得到的导数
\(\frac{\partial f}{\partial \vec l}=lim_{\rho \to 0} \frac{f(x_0+\Delta x,y_0+\Delta y,z_0+\Delta z)-f(x_0,y_0,z_0)}{\rho}. \)
\(\rho = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}\)
\((\Delta x,\Delta y,\Delta z)=\rho(\cos\alpha,\cos\beta,\cos\gamma)\)

   若在该点可微，$\nabla f$ 是梯度向量，则
若在该点可微,$\frac{\partial f}{\partial \mathbf{u}} = \nabla f \cdot \mathbf{u}$
---
2. 偏导数
$$\frac{\partial f}{\partial x} = \lim_{h\to 0} \frac{f(x+h,y)-f(x,y)}{h}$$
$$\frac{\partial f}{\partial y} = \lim_{h\to 0} \frac{f(x,y+h)-f(x,y)}{h}$$
---
3. 全微分
$$df = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy$$
---
4. 多元函数极值
   - **驻点**
   $\begin{cases}
   \frac{\partial f}{\partial x} = 0 \\
   \frac{\partial f}{\partial y} = 0
   \end{cases}$
   - **充分条件**
   $$A=f'_{xx},B=f'_{xy},C=f'_{yy}$$
   $$B^2 - AC < 0$$
   - **Lagrange乘数法**
   $$L(x,y,\lambda) = f(x,y)+\lambda g(x,y)$$
   条件：
   $$\begin{cases}
   \frac{\partial L}{\partial x} = 0 \\
   \frac{\partial L}{\partial y} = 0 \\
   \frac{\partial L}{\partial \lambda} = 0
   \end{cases}$$
---
5. 梯度
   $$\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right)$$
   - 梯度的几何意义：梯度指向函数增长最快的方向，且其模长表示增长率。

## 重积分
1. 换元法

2. 

## 曲线积分
1. 第一类曲线积分
    - **直接投影法**
    $$L=\int_a^b f(x) \sqrt{1+\left(\frac{dz}{dx}\right)^2} \, dx$$
    - **参数化法**
    $$L=\int_a^b \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2 + \left(\frac{dz}{dt}\right)^2} \, dt$$
---
2. 第二类曲线积分
    - **直接投影法**
    $$L=\int_a^b \mathbf{F} \cdot d\mathbf{r} = \int_a^b (F_1 \frac{dx}{dt} + F_2 \frac{dy}{dt} + F_3 \frac{dz}{dt}) \, dt$$
    - **参数化法**
    $$L=\int_C \mathbf{F} \cdot d\mathbf{r} = \int_C (F_1 dx + F_2 dy + F_3 dz)$$
---
3. 格林公式(平面区域)
   $$\iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dx dy=\oint_L Pdx + Qdy$$
---
4. 路径无关
   - 条件：
      - $$\frac{\partial Q}{\partial x} = \frac{\partial P}{\partial y}$$
   - 关键点:
      - 取路径$$(x_1,y_1) \to (x_2,y_2)$$为$$(x_1,y_1) \to (x_2,y_1) \to (x_2,y_2)$$
      - 复连通区域更换路径方便参数化

## 曲面积分
1. 第一类曲面积分
   - **直接投影法**
    $$S=\iint_D \mathbf{F} \cdot d\mathbf{S} = \iint_D (F_1 dx + F_2 dy + F_3 dz)$$
    - **参数化法**
    $$S=\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S (F_1 dx + F_2 dy + F_3 dz)$$
---
2. 第二类曲面积分
   - **直接投影法**
    $$S=\iint_D f(x,y) \sqrt{1+\left(\frac{\partial z}{\partial x}\right)^2+\left(\frac{\partial z}{\partial y}\right)^2} \, dx \, dy$$
   - **参数化法**
      $$S=\iint_D \sqrt{\left(\frac{\partial x}{\partial u}\right)^2 + \left(\frac{\partial y}{\partial u}\right)^2 + \left(\frac{\partial z}{\partial u}\right)^2} \, du \, dv$$
   - **格林公式(平面区域)**
      $$S=\iint_D \left( \frac{\partial F_2}{\partial x} - \frac{\partial F_1}{\partial y} \right) dx dy=\oint_L F_1dx+ F_2dy$$
---
3. 高斯公式(闭合曲面)
   $$\iiint_V (\frac{\partial F_3}{\partial x}+\frac{\partial F_2}{\partial y}+\frac{\partial F_1}{\partial z})    \cdot d\mathbf{S} = \oiint_S F_1dxdy+F_2dxdz+F_3dydz$$
---
4. 斯托克斯公式
   $$\iint_S \mathbf{F} \cdot d\mathbf{S} = \oint_C \mathbf{F} \cdot d\mathbf{r}$$
---
5. 场论初步
   - **向量场**
   $$\mathbf{F} = (F_1, F_2, F_3)$$
   - **散度**
   $$\nabla \cdot \mathbf{F} = \frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} + \frac{\partial F_3}{\partial z}$$
   - **旋度**
   $$\nabla \times \mathbf{F} = \left( \frac{\partial F_3}{\partial y} - \frac{\partial F_2}{\partial z}, \frac{\partial F_1}{\partial z} - \frac{\partial F_3}{\partial x}, \frac{\partial F_2}{\partial x} - \frac{\partial F_1}{\partial y} \right)$$