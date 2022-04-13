
## Add Mutual Information Loss for VFL

### VFL setting
1. Active party and passive party each calculate $H_a=M_a(X_a;{\theta}_a)$ and $H_p=M_p(X_p;{\theta}_p)$.
2. Passive party sends $[[H_p]]$ to active party.
3. Active calculates $H=U(H_a,[[H_p]])$.
4. $\mathcal{L}=-I(H,Y)+\lambda I(H_p,Y)$. By minimizing $\mathcal{L}$, we can minimize $I(H_p,Y)$ and maximize $I(H,Y)$, in order to make $H_p$ contain less information about $Y$ while $H$ contains more information about $Y$.

### Calculation of the loss function
* We have: $\mathcal{L}=-I(H,Y)+\lambda I(H_p,Y)$.
* $\begin{aligned}KL(P||Q)=H(P,Q)-H(P)\end{aligned}$
* $\begin{aligned}I(P,Q)&=H(P)+H(Q)-H(P,Q)\\&=H(P)-KL(Q||P)\\&=H(Q)-KL(P||Q)\end{aligned}$
* $\begin{aligned}\mathcal{L}&=-I(H,Y)+\lambda I(H_p,Y)\\&=-[H(Y)-KL(H||Y)]+\lambda [H(Y)-KL(H_p||Y)]\\&=(\lambda -1)H(Y)+KL(H||Y)-\lambda KL(H_p||Y)\end{aligned}$
* Since $H(Y)$ is deterministic, then $\mathcal{L}=KL(H||Y)-\lambda KL(H_p||Y)$.
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
