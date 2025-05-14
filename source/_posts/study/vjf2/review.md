---
title: 期中考复盘
date: 2025-04-14 15:00:00
---

$$1.在\Delta ABC中,\angle C=90^\circ,边BC,CA,AB的长分别为a,b,c,请用向量的方法证明勾股定理：c^2=a^2+b^2\\$$
$$\overrightarrow {AB}=\overrightarrow {AC}+\overrightarrow {CB},|AB|=|AC+CB|\Rightarrow c^2=a^2+b^2-2abcosC=a^2+b^2$$

$$2.已知平行四边形ABCD中,两邻边BC,CD的中点分别为E,F.向量AE=i-3j+2k,AF=5i+2j+3,求平行四边形ABCD的面积.$$
$$S_{\Delta AEF}=\frac12AE\times AF=\frac{3}{8}S_{ABCD}=\frac{52\sqrt3}{3},利用比例简化计算,无需具体求出AB,AC$$
    
$$3.设直线L1:\frac{x+2}{1}=\frac{y-3}{-1}=\frac{z+1}{1},L2:\frac{x+4}{2}=\frac{y}{1}=\frac{z-4}{3}.试求与L_1,L_2都垂直且相交的直线方程.$$
$$\vec v_1=(1,-1,1),\vec v_2=(2,1,3)\Rightarrow \vec v=\vec v_1\times \vec v_2=(-4,-1,3).\\
设L_1公共点P,L_2上公共点Q\Rightarrow \overrightarrow {PQ}\cdot \vec v_1=0,\overrightarrow {PQ}\cdot \vec v_2=0,解得s=0.$$

$$4.求直线L_1\frac{x+2}{1}=\frac{y}{1}=\frac{z-1}{-1}在平面π：x-y+2z-1=0上的投影直线l_0的方程，并求直线l_0绕y轴旋转一周所成曲面的方程.$$
$$选择直线L_1上点A(-2,0,1),B(0,2,-1),求出在平面上的投影点A',B',A'B'即为l_0.\\
平面法向量\vec n=(1,-1,2),投影点公式：P'=P - [\frac{ax₀ + by₀ + cz₀ -d}{a² + b² + c²}](a,b,c)\\
投影点A'和B'的坐标分别为A'(-\frac{11}{6},-\frac{1}{6},\frac{4}{3}),B'(-\frac{1}{2},\frac{1}{2},1)\Rightarrow l_0:\frac{x+\frac{1}{2}}{4}=\frac{y-\frac{1}{2}}{2}=\frac{z-1}{-1}.\\
设直线l_0上一点P(x_0,y_0,z_0),旋转后变为P'(x,y,z)\\
\left\{\begin {matrix}
x^2+y^2=x_0^2+z_0^2\\
y=y_0\\
\frac{x+\frac{1}{2}}{4}=\frac{y-\frac{1}{2}}{2}=\frac{z-1}{-1}
\end{matrix}\right.
$$

$$5.设z=yf(xy,x-y)+g(x+y),其中f具有二阶连续的偏导数,g具有二阶导数,求:\frac{\partial z}{\partial x},\frac{\partial^2 z}{\partial x\partial y}.$$

$$6.设z=z(x,y)是由方程yz^3-xz^4+z^5=1所确定的隐函数,求:\frac{\partial z}{\partial x}|_{(0,0)},\frac{\partial^2 z}{\partial x^2}|_{(0,0)}.$$

$$7.叙述函数z=f(x,y)在点P_0(x_0,y_0)处的可微的定义,并研究函数f(x,y)=\sqrt{|xy|}在原点处的可微性.$$
$$函数z=f(x,y)在点P_0(x_0,y_0)处可微的定义是:存在常数A和B,使得当自变量增量\Delta x和\Delta y趋近于零时，函数的全增量可表示为dz=Adx+Bdy+\circ((\Delta x)^2+(\Delta y)^2),其中余项满足当\rho\rightarrow0时,\frac{\circ\rho}{\rho}\rightarrow0.此时，A和B分别为函数f在点P_0处对x,y的偏导数.\\
用极限形式表达,即lim_{(dx,dy)\rightarrow(0,0)}\frac{f(x_0+dx,y_0+dy)-f_x(x_0,y_0)-f'_x(x_0,y_0)dx−f'_y(x_0,y_0)dy}{\sqrt{(dx)^2+(dy)^2}}=0.\\
当上述条件满足时，称函数f在点P_0处可微,其全微分为dz=f'_x(x_0,y_0)dx+f'_y(x_0,y_0)dy.$$
    
$$8.若变换\left\{\begin{matrix}u=x+ay\\v=x+by\end{matrix}\right.可将微分方程\frac{\partial^2z}{\partial x^2}+4\frac{\partial^2 z}{\partial x\partial y}+3\frac{\partial^2 z}{\partial y^2}=0简化为关于u,v的微分方程\frac{\partial^2z}{\partial u\partial v}=0,其中z具有二阶连续偏导数,求常数a,b的值.$$
    1+4a+3a²=0
    2+a+b+2ab=0
    1+4b+3b²=0


