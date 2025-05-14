---
title: 期末考知识点查漏
date: 2025-04-14 15:00:00
---

## 1. 幂级数

### 重要函数和延申：
1. \( S(x) = \sum_{n=0}^{\infty} x^n = \frac{1}{1-x}, \quad |x| < 1 \)
2. \( T(x) = \sum_{n=1}^{\infty} \frac{x^n}{n} = -\ln(1-x), \quad |x| < 1 \)

凑 \( S(x) \)，例如：
\( T(x)' = S(x) = \frac{1}{1-x}, \quad T(x) = \int_0^\infty \frac{1}{1-x} \, dx = -\ln(1-x) \)

### 泰勒展开：
3. \( \sum_{n=1}^{\infty} \frac{x^{2n}}{(2n)!} = \cosh(x), \quad \text{for all } x \)
4. \( \sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x, \quad \text{for all } x \)
5. \( \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = \sin(x), \quad \text{for all } x \)
6. \( \sum_{n=0}^{\infty} \frac{x^{2n}}{(2n)!} = \cos(x), \quad \text{for all } x \)
7. \( \sum_{n=1}^{\infty} \frac{x^n}{n!} = \ln(1+x), \quad |x| < 1 \)

## 2. 傅里叶级数

\( f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right] \)

其中：
\[
a_n = \frac{2}{T} \int_{t_0}^{t_0+T} f(x) \cos(n\omega x) \, dx, \quad
b_n = \frac{2}{T} \int_{t_0}^{t_0+T} f(x) \sin(n\omega x) \, dx
\]

### 注意延拓方式

#### 狄利克雷收敛定理：
1. 连续或有限个间断点
2. 有限个极值点

连续：\( f(x) \)

间断：\( \frac{1}{2} \left[ f^{-}(x) + f^{+}(x) \right] \)

## 3. 方向导数
定义:函数定义域的内点对某一方向求导得到的导数
\(\frac{\partial f}{\partial \vec l}=lim_{\rho \to 0} \frac{f(x_0+\Delta x,y_0+\Delta y,z_0+\Delta z)-f(x_0,y_0,z_0)}{\rho}. \)
\(\rho = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}\)
\((\Delta x,\Delta y,\Delta z)=\rho(\cos\alpha,\cos\beta,\cos\gamma)\)
### 方向导数的几何意义
方向导数表示函数在某一特定方向上的变化率。
## 4. 曲线积分与曲面积分