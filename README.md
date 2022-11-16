# Texture Synthesis

Ce repository s'appuie sur l'article *Texture Optimization for Example-based Synthesis* écrit par Vivek Kwatra, Irfan Essa, Aaron Bobick et Nipun Kwatra (https://www.cc.gatech.edu/cpl/projects/textureoptimization/#:~:text=Texture%20Optimization%20for%20Example%2Dbased%20Synthesis&text=We%20present%20a%20novel%20technique,to%20a%20given%20input%20sample.). Cet article publié en 2005 proprose une méthode de génération de texture basée sur la minimisation d'une énergie globale grâce à un algorithme similaire à l'algorithme EM. 

La méthode proposée par l'article repose sur une approche globale basée sur une mesure de similarité entre les voisinages de l'Input et les voisinages de la texture en cours de synthèse. On note $X$ la texture que l'on synthétise et $Z$ la texture originale. De plus on appelle $x_p$ le vecteur contenant les valeurs des pixels du voisinage centré en $p$ dans l'image X et $z_p$ le voisinage de Z le plus proche de $x_p$. L'objectif est de minimiser l'énergie suivante qui est la somme des énergies pour chaque voisinage de X :
\begin{equation}
E_t = \sum_{p\in X_{vois}} || x_p - z_p ||^2
\end{equation}
Cette optimisation se fait successivement par rapport aux $x_p$ et aux $z_p$. Par ailleurs, on considère seulement certains voisinages de X qui seront de taille $\omega$ et séparés de $e \sim \omega/4$ pixels. 

L'optimisation consiste à itérer successivement 2 étapes de minimisation de l'énergie par rapport aux $x_p$ et aux $z_p$. Ces deux étapes rappellent fortement le format de l'algorithme EM (avec $x_p$ les variables et $z_p$ les paramètres) :

**E_step :**

Cette étape correspond à la minimisation de l'énergie par rapport aux voisinages $x_p$ de l'image X. Concrètement cela revient à donner aux $x_p$ la valeur des pixels dans $z_p$ en faisant une moyenne aux endroits où les voisinages overlapent. 
On commence par créer un masque avec la fonction create_overlap_mask qui indique le nombre de voisinages qui overlapent à chaque pixel de l'image. Puis chaque E_step consiste à remplir l'image X avec les valeurs des voisinages $z_p$ (en sommant aux endroits où il y a recouvrement) puis à diviser par le masque. 

**M_step :**

L'objectif de cette étape est de minimiser l'énergie par rapport aux $z_p$, ce qui consiste à trouver le voisinage de Z le plus proche de chaque $x_p$. Pour accélérer cette recherche, l'article suggère d'utiliser un clustering hiérarchique et de construire un arbre avec les voisinages de Z. On procède donc à la construction de cet arbre au début de l'algorithme grâce à la fonction KDTree du module sklearn.neighbors. Puis à chaque M_step on utilise cet arbre et la fonction query pour trouver les $z_p$ et on stocke les indices de ces voisinages dans la liste $Z_p$.

