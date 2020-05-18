# Mon pipeline pour l'identification d'espèces sur photos

J'ai un ensemble de 46 photos annotées à la main. L'information sur ce qui a été détecté dans chaque photo apparait dans les métadonnées des photos. Sous Mac, il suffit de faire un Cmd + I pour avoir cette info. Les photos sont stockées dans un dossier pix/ dont le chemin absolu est /Users/oliviergimenez/Desktop/. 

Je voudrais évaluer les performances (vrais positifs, faux négatifs et faux positifs) du modèle entrainé par Gaspard à reconnaître les espèces qui sont sur ces photos, et en particulier lynx, chamois et chevreuils. 

Ci-dessous on trouvera les différentes étapes du pipeline. C'est un mix de scripts R et Python. On applique une procédure en 2 étapes, détection puis classification. L'idée est aussi appliquée par d'autres pour des projets (et avec des moyens) beaucoup plus ambitieux, voir par exemple [celui-ci](https://medium.com/microsoftazure/accelerating-biodiversity-surveys-with-azure-machine-learning-9be53f41e674). 

Le gros du boulot (en particulier l'entrainement d'un modèle de classification, cf. étape 4) a été fait par Gaspard Dussert en stage en 2019 avec Vincent Miele. Plus de détails [sur le site dédié du GdR EcoStat](https://ecostat.gitlab.io/imaginecology/). 

## Etape 1. Redimensionnement.

On redimensionne d'abord les images. Pour ce faire, on applique ces quelques lignes de code dans R. On utilise le package magical qui appelle l'excellent petit logiciel imagemagick. Les photos contenues dans le répertoire /Users/oliviergimenez/Desktop/pix sont redimensionnées en 1024x1024 dans le répertoire /Users/oliviergimenez/Desktop/pix_resized. Le nom de chaque photo est affublé d'un resized pour les différentier des photos originales. 

```
# load package to make R talk to imagemagick
library(magick)

# where the pix to resize are
folder <- "/Users/oliviergimenez/Desktop/pix/"

# where the pix, once resized, should be stored
dest_folder <- "/Users/oliviergimenez/Desktop/pix_resized/"

# create directory to store resized pix
dir.create(dest_folder)

# list all files in the directory with pix
file_list <- list.files(path = folder)

# resize them all !
for (i in 1:length(file_list)){
	pix <- image_read(paste0(folder,file_list[i]))
	pixresized <- image_resize(pix, '1024x1024')
	namewoextension <- strsplit(file_list[i], "\\.JPG")[[1]]
	image_write(pixresized, paste0(dest_folder,namewoextension,'resized.JPG'))
}
```

## Etape 2. Détection. 

On fait la détection des objets dans les photos. On utilise [MegaDetector](https://github.com/microsoft/CameraTraps#overview) pour se faciliter la vie. Cet algorithme va détecter les objets sur les photos et leur associer un cadre, une boîte. 

Pour ce faire, il faut d'abord télécharger [CameraTraps](https://github.com/microsoft/CameraTraps). Puis, depuis un Terminal, se mettre dans le répertoire CameraTraps/ et suivre [les instructions d'installation](https://github.com/microsoft/CameraTraps#initial-setup). Si on n'a pas de GPU, il faut modifier le fichier dans le fichier environment-detector.yml en commentant la ligne 
```
tensorflow-gpu>=1.9.0, <1.15.0
```
en
```
# tensorflow-gpu>=1.9.0, <1.15.0
```
et ajouter la ligne
```
tensorflow=1.14
```

Il se peut qu'il faille installer des modules, dans ce cas, utiliser pip install dans le Terminal. 

Ensuite, dans le Terminal, faire 
```
conda init
```
puis
```
conda activate cameratraps
```

Avant de se lancer, il faut récupérer le modèle megadetector_v3.pb pour la détection [ici](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb). 

On est prêt à utiliser MegaDetector. Trois options s'offrent à nous. 

a. On traite une seule photo, disons '1.3 D (145)resized.JPG', et on lui met un cadre là où un objet est détecté. Taper dans le Terminal : 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_file /Users/oliviergimenez/Desktop/pix_resized/1.3\ D\ \(145\)resized.JPG
```

Le traitement prend quelques secondes. Un cadre a été ajouté sur la photo traitée, ainsi que la catégorie de l'objet détecté et un degré de confiance. 

b. On traite toutes les photos du dossier pix_resized. 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_dir /Users/oliviergimenez/Desktop/pix_resized/
```

Les photos avec cadre sont ajoutées dans le même répertoire pix_resized, leur nom est juste modifié avec l'ajout de 'detections' pour signifier qu'elles ont été traitées. On remarque que pour la photo déjà traitée au a., un autre cadre a été ajouté, et la photo a été doublée. Si pour les animaux, le taux de succès est de 100%, pour les véhicules, il est de 0%, et pour les humains ce taux est élevé, mais pas de 100%. Pour les photos vides, pas de faux positifs, ie pas de cadre là où pas d'objets. Vu notre objectif, celui de travailler sur les interactions entre lynx, chamois, et chevreuils, le fait de détecter tous les animaux, et de ne pas mettre des cadre là où il n'y a pas d'animaux, nous semble prometteur, et ok pour continuer.

c. On ne touche pas aux photos, on crée un fichier json dans lequel on récupère les coordonnées des cadres, ainsi que les catégories des objets détectés. 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector_batch.py /Users/oliviergimenez/Desktop/megadetector_v3.pb /Users/oliviergimenez/Desktop/pix_resized/ /Users/oliviergimenez/Desktop/box_pix.json
```
Attention, avant de lancer la commande au-dessus, transférer les 47 photos avec objets encadrés obtenus aux points a. et b. Dans un autre répertoire, par exemple pix_resized_detections, sinon l'étape c. portera sur les 93 photos.

On peut ouvrir le fichier json ainsi créé avec un éditeur de texte. On peut voir des blocs, un bloc correspondant au traitement d'une image, par exemple : 

``` {
   "file": "/Users/oliviergimenez/Desktop/pix_resized/I__00016 (6)resized.JPG",
   "max_detection_conf": 0,
   "detections": []
  },
  {
   "file": "/Users/oliviergimenez/Desktop/pix_resized/Cdy00020resized.JPG",
   "max_detection_conf": 0.999,
   "detections": [
    {
     "category": "1",
     "conf": 0.999,
     "bbox": [
      0.6317,
      0.5045,
      0.3677,
      0.1947
     ]
    }
   ]
  },
```

On a le nom de l'image, la catégorie de l'objet détecté (0 = vide, 1 = , 2 = , 3 = ), le degré de confiance (conf) ainsi que les caractéristiques de la boîte associée à l'objet (xmin, ymin, width, height). On arrangera ce fichier à l'étape d'après pour en extraire l'information pertinente.

3 On crée le fichier test.csv avec scripts R. 

4. On prédit.

5. On évalue les performances avec script R. 



/Users/oliviergimenez/Desktop/DLcameratraps/keras-retinanet-master/keras_retinanet/bin/evaluate.py --convert-model --save-path test_pred/ --


















(4. Faire la prédiction avec le modèle entrainé sur pix Jura ; https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet/). Voir plus bas pour plus de détails. Le modèle déjà entrainé est ici https://mycore.core-cloud.net/index.php/s/Prj6xeu0GqNWaXB.

(5. Évaluer les performances TP, FN, FP avec script R postprocessML.R)


Concernant l'étape 2 :

Se mettre dans le répertoire où se trouent les script environment.yml et environment-detector.yml
Faire 
conda env create --file environment.yml
Puis faire
conda env create --file environment-detector.yml
En changeant si besoin 
- tensorflow-gpu>=1.9.0, <1.15.0
En
- tensorflow=1.14

Bien pensé à activer conda via un :
conda activate cameratraps

Et avant ça, aller dans le répertoire CameraTraps et faire un 
conda env create --file environment.yml
puis un 
conda init

Suite au message d'erreur AttributeError: module 'tensorflow' has no attribute 'GraphDef', j'ai downgrade tensorflow via
python3 -m pip install tensorflow==1.14, et la tout roule

On est alors prêt a utiliser megadetector :

Soit on met des cadres sur les photos via (5 secondes par pix en moy):

Pour une image, python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_file /Users/oliviergimenez/Desktop/36.2_G_Lot3resized/I__00001\ \(7\)resized.JPG

Pour toutes les images d'un répertoire 
python3 /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_dir /Users/oliviergimenez/Desktop/36.2_G_Lot3resized/

Dans le répertoire où se trouvent les photos, vous trouverez une copie de chacune d'entre elles, avec ajout de 'detections' dans le nom du fichier, et si vous ouvres les photos, il y a un cadre avec animal, person ou vehicule. Et pas de cadre dans les photos où rien n'est trouvé.

Soit on crée un fichier json avec les coordonnées des boites
python3 /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector_batch.py /Users/oliviergimenez/Desktop/megadetector_v3.pb /Users/oliviergimenez/Desktop/36.2_G_Lot3resized/ /Users/oliviergimenez/Desktop/36.2_G_Lot3resized/box_ain.json


Concernant l'étape 4.

0. Follow Step 2 and Installation of RetinaNet https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet#2-installation-of-retinanet

1. Download directory https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet

2. Follow https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet#6-detecting-the-animals-on-a-test-set

2a. Download trained model ftp://pbil.univ-lyon1.fr/pub/datasets/imaginecology/retinanet/retinanet_tutorial_best_weights.h5 and put it in directory

2b. Get pictures on which do the detection by typing in a terminal
wget -r -nd -np ftp://pbil.univ-lyon1.fr/pub/datasets/imaginecology/retinanet/test/ and put them in the test/ directory in the main directory

3. Run evalation algo w/ command /Users/gimenez/Desktop/keras-retinanet-master/keras_retinanet/bin/evaluate.py --convert-model --save-path test_pred/ --score-threshold 0.5 csv test.csv class.csv retinanet_tutorial_best_weights.h5

In case you get ModuleNotFoundError: No module named 'keras_retinanet.utils.compute_overlap', run the command python setup.py build_ext --inplace in the directory where the setup.py file is.


Pour afficher à l'écran (dans le Terminal) le nom de la photo, l'espèce détectée, la précision, et les coordonnées de la boîte, voilà un script Python écrit par Vincent Miele le 11 mai 2020 :

"J'ai ajouté un script Python décrit en bas de la page
https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet

Il s'agit de "detect2txt.py" :
-il te faut faire un "git pull"
-il te faut éditer le fichier et modifier les 3 paramètres (ceux qui
sont par défaut marchent si tu as suivi le tuto)
-et puis lancer "python3 detect2txt.py"" ; chez moi il faut ajouter le chemin absolu python3 /Users/oliviergimenez/Desktop/DLcameratraps/keras-retinanet-master/keras_retinanet/bin/detect2txt.py et avoir au préalable installer deux librairies manquantes, matplotlib et pandas

Concernant l'étape 5, Gaspard fournit 2 scripts detect.py et detect2.py en pj. La différence entre les deux fichiers est dans la façon de calculer les faux négatifs. Le script detect.py calcule TP, FN et FP alors que detect2.py permet d'aller un peu plus dans le détail et de séparer les faux négatifs en FNvoid si l'animal n'est pas détecté et FN_false si l'animal a été détecté mais mal classifié. Ça fait suite à votre idée d'ajouter dans le background les classes avec peu de photos pour réduire le nombre de faux positifs. 

Note : le calcul de ces métriques d'erreur ne tiennent pas compte des erreurs faites à l'étape de la détection des objets avec MegaDetector. 
