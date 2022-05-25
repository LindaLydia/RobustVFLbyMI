
## Add Mutual Information Loss for VFL

### VFL setting
1. Active party and passive party each calculate $H_a=M_a(X_a;{\theta}_a)$ and $H_p=M_p(X_p;{\theta}_p)$.
2. Passive party sends $[[H_p]]$ to active party.
3. <font color=Red>Active party generates $Z$ according to $[[H_p]]$ and use $\color{red}{\mathcal{L}_{info}=I([[H_p]],Z)-I(Z,Y)}$ to make $Z$ contains more information about $Y$ and $[[H_p]]$ contains less information about $Z$ thus it contains less information about $Y$.</font>
4. Active party calculates $\color{red}{\hat{Y}=H=S(H_a,Z)}$ instead of $\hat{Y}=H=U(H_a,[[H_p]])$ where $S$ is a non-linear function like softmax of summation.
5. ~~$\mathcal{L}=-I(H,Y)+\lambda I(H_p,Y)$. By minimizing $\mathcal{L}$, we can minimize $I(H_p,Y)$ and maximize $I(H,Y)$, in order to make $H_p$ contain less information about $Y$ while $H$ contains more information about $Y$.~~ <font color=Red>No, with $Z$, we use $\mathcal{L}=CrossEntropy(H,Y)+\mathcal{L}_{info}$ as the final loss of the VFL model.</font>

### Calculation of the loss function
* We have: $\mathcal{L}=-I(H,Y)+\lambda I(H_p,Y)$.
* $\begin{aligned}KL(P||Q)=H(P,Q)-H(P)\end{aligned}$
* $\begin{aligned}I(P,Q)&=H(P)+H(Q)-H(P,Q)\\&=H(P)-KL(Q||P)\\&=H(Q)-KL(P||Q)\end{aligned}$
* (1) 
    * $\begin{aligned}\mathcal{L}&=-I(H,Y)+\lambda I(H_p,Y)\\&=-[H(Y)-KL(H||Y)]+\lambda [H(Y)-KL(H_p||Y)]\\&=(\lambda -1)H(Y)+KL(H||Y)-\lambda KL(H_p||Y)\end{aligned}$
    * Since $H(Y)$ is deterministic, then $\mathcal{L}=KL(H||Y)-\lambda KL(H_p||Y)$.
* (2) 
    * $\begin{aligned}\mathcal{L}&=-I(H,Y)+\lambda I(H_p,Y)\\&=-[H(H)-KL(Y||H)]+\lambda [H(H_p)-KL(Y||H_p)]\end{aligned}$
    <!-- * $$\color{red}{}$$ -->
    * CrossEntropy loss is a special version of KL-Divergence loss. So, we rewirite (2) as $\mathcal{L}=-[H(H)-CrossEntropy(H,Y)]+\lambda [H(H_p)-CrossEntropy(H_p,Y)]$
    * <font color=Dandlion>Experiments show that, this loss function is not suitable for the main task. When $\lambda >0$ and when minimizing the loss, $[KL(Y||H_p)]$ will be pushed to positive infinity which result in $loss\approx -1e6$ even when $\lambda=0.01$ and hurts the main task accuracy to a great extent.</font>
        * <font color=Orange>Possible solution1:(maybe not right)</font> Reparameterization. Which means, we calculate the distribution of $H_p \sim N(\mu,\sigma)$ and make it more close to $N(0,1)$ using $L_{sim}=-log\sigma + \frac{1}{2}(\sigma^2+\mu^2-1)$ to replace $[H(H_p)-KL(Y||H_p)]$ and gets the final loss as $\color{orange}{\mathcal{L}=-[H(H)-KL(Y||H)]+\lambda(-log\sigma + \frac{1}{2}(\sigma^2+\mu^2-1))}$
        * <font color=Orange>Possible solution2:</font> We want $KL(Y||H_p)$ as large as possible. But our final goal is to make $H_p$ contain as less information about $Y$ as possible. So, we can push $H_p$ as close to "equal probability" as possible. That is to make the final loss as $\color{orange}{\mathcal{L}=-[H(H)-KL(Y||H)]+\lambda CrossEntropy(H_p, [1/class]*class)}$
        * <font color=Orange>Possible solution3:</font> Stop the training process when the loss is less than a positive threshold, like $0.5$ just to give an example casually.
* `torch.nn.KLDivLoss` is a kind of loss provided by pytorch.




## Reference for Latex Equations
1. $\mathbb{R}$
2. $\mathcal{R}$
3. $\mathscr{R}$
4. $\mathrm{R}$
5. $\mathbf{R}$
6. $\mathit{R}$
7. $\mathsf{R}$
8. $\mathtt{R}$
9. $\mathfrak{R}$

* [Reparameterization](https://spaces.ac.cn/archives/6705)
