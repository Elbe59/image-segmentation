# 8INF804 Traitement Numérique des Images : TP n°2 "Segmentation d'une image"

## Description
Ce projet a pour objectif la segmentation de différentes images d'échantillons de roche afin d'identifier un à un les grains et calculer la moyenne de chacun des channels BGR du grain pour déterminer le type de minéral dont il s'agit.

## Utilisation (sous windows)
- Rendez-vous dans le répertoire principal : cd ./tp2_8inf804
- En utilisant notre environnement virtuel :
  - la commande suivante permet d'exécuter le programme : .\venv\Scripts\python.exe .\main.py .\Images\
- En utilisant votre environnement :
  - Installer d'abord les librairies requises avec la commande : pip install -r requirements.txt
  - Ensuite utiliser votre interpréteur python pour lancer le script main.py avec comme argument le chemin d'accès vers le dossier d'images.
- Il est possible de changer de répertoire en remplaçant le dossier d'image

## Visualisation et sauvegarde des résultats
Après avoir exécuté le programme avec la commande ci dessus, les résultats s'afficheront un par un sur une fenêtre matplotlib
contenant 4 visuels, l'image d'origine, l'image après prétraitement, le local max de chaque segment et pour finir l'image segmentée avec une coloration de chaque segment pour une meilleure visualisation.
Pour finir les résultats finaux avec tous les segments identifiés avec un contour jaune se trouvera dans le dossier Output/Images
Et les Dataframes correspondant se trouveront dans le dossier Output/Dataframes




## Auteurs
- Ilyes Bennaga (BENI26099904)
- Quentin Callens (CALQ06109900)
- Pierre Lefebvre (LEFP18039809)

![](ressources/GUI/ninja_2.png)![](ressources/GUI/ninja_3.png)![](ressources/GUI/ninja_4.png)![](ressources/GUI/ninja_1.png)