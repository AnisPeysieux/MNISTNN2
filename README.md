# Neural-network-for-image-recognition

Ce programme cree un reseau de neurones de type perceptron multicouche avec 784 neurones en couche d'entree, deux couches de 200 neurones en couches cachees et 10 neurones en couche de sortie.
Il effecture 35 iterations sur la base de donnee d'entrainement MNIST (reconnaissance de caractere numeriques) avant d'effectuer un test sur la base de donnees de test.

Compilation:
  Effectuer les commande suivantes:
    -mkdir build
    -cd build
    -cmake ..
    -make install
  L'executable se trouve maintenant dans le repertoire bin/ de la racine du projet.

La documentation du code source se trouve dans doc/html/index.html.
Le repertoire graphs/ contient des sous-repertoire donc le nom correspond a un certain nombre d'iterations. Dans chacun de ces repertoires se trouve un ensemble de graphiques du taux de reussite de differents reseaux apres chaque iteration. Les nombres indiquees dans les de fichiers des grahiques correpondent a la configuration du reseau. Par exemple "300_300_300_10" dans le repertoire 100 correspond a un reseau qui contient 3 couches cachees de 300 neurones et une couche de sortie de 10 neurones. La couche d'entree contient dans tous les cas 784 neurones
