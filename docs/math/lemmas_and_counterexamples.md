# Thermal solution and global curvature

Mhamed Halhoul.

This is a distinct documentary excerpt of the saved mathematical audit. Section L01 below is reproduced verbatim, with its original hypotheses. The [report](../../report/main.pdf) contains the study and its numerical limitations. This analytic argument does not certify total floating-point errors.

<a id="l01"></a>
## L01 — Solution thermique, existence et courbure globale

**Statut : démontré pour le modèle exact du brief.**

Pour t≥0 et 0≤x≤1, poser
\[
u(x,t)=e^{-\pi^2t}\sin(\pi x)
       +\tfrac15e^{-4\pi^2t}\sin(2\pi x).
\]
Chaque terme \(A_ke^{-(k\pi)^2t}\sin(k\pi x)\) a
\[
\partial_t=- (k\pi)^2A_ke^{-(k\pi)^2t}\sin(k\pi x)
=\partial_{xx}.
\]
La somme vérifie donc \(u_t=u_{xx}\). Aux deux bords les sinus sont nuls ; à t=0 la somme vaut \(\sin(\pi x)+\tfrac15\sin(2\pi x)\), exactement la condition initiale. Le signe du second mode est « + », tel que lu. Il n'y a aucune correction d'un signe historique.

Dans la classe des solutions classiques suffisamment régulières pour l'intégration par parties, deux solutions de mêmes données ont une différence v avec données nulles. Alors
\[
\frac12\frac d{dt}\int_0^1v^2\,dx
=\int_0^1v\,v_{xx}\,dx=-\int_0^1v_x^2\,dx\leq0.
\]
L'énergie initiale est nulle et reste nulle : unicité dans cette classe. Cette conclusion ne suppose ni réseau ni solveur numérique.

Au capteur x=1/4 :
\[
F(p)=T(p)=Ae^{-cp}+Be^{-dp},
\quad A=1/\sqrt2,\ B=1/5,\ c=\pi^2,\ d=4\pi^2.
\]
Les dérivées sont
\[
F'=-Ac e^{-cp}-Bd e^{-dp}<0,\quad
F''=Ac^2e^{-cp}+Bd^2e^{-dp}>0,\quad
F'''=-Ac^3e^{-cp}-Bd^3e^{-dp}<0.
\]
Ainsi F'' décroît, et, sur tout P,
\[
0<\mu=m=F''(b)
=\frac{\pi^4}{\sqrt2}e^{-\pi^2/5}
+\frac{(4\pi^2)^2}{5}e^{-4\pi^2/5}
\leq F''(p)\leq F''(a)=L_{\rm phys}<\infty.
\]

Pour \(J_\lambda(p)=F(p)+\lambda p\), \(J'=F'+\lambda\) et \(J''=F''\). Pour tous x,y∈P, en intégrant la dérivée seconde,
\[
J(y)\geq J(x)+J'(x)(y-x)+\frac m2(y-x)^2.
\]
C'est une forte convexité **globale sur P**, avec m exact strictement positif, indépendant de λ. Il n'existe ici ni observation bruitée y ni terme de résidu influant sur m. Les amplitudes, capteur, diffusivité et domaine sont fixés ; la conclusion ne se transfère pas à une modification de ces données.

J est continu sur le compact P : un minimiseur existe. La forte convexité le rend unique. Poser \(q_a=-F'(a)\), \(q_b=-F'(b)\). Comme F' croît strictement, \(q_a>q_b>0\).
- λ≥q_a : J'≥0 sur P, \(p^\star=a\).
- λ≤q_b : J'≤0 sur P, \(p^\star=b\).
- \(q_b<\lambda<q_a\) : unique racine intérieure de J'=0.

Les égalités sont incluses dans les cas de bord. La plage du brief \([q_b/2,3q_a/2]\) est bien ordonnée, strictement positive, et inclut les trois régimes. La convexité en p ne porte pas sur les poids θ.

