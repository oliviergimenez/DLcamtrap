# Piégeage photo et pipeline pour l'identification d'espèces

On a un ensemble de 46 photos annotées à la main téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/ub5iTNSktszLvCv). L'information sur ce qui a été détecté dans chaque photo apparait dans les métadonnées des photos. Sous Mac, il suffit de faire un Cmd + I pour avoir cette info. Les photos sont stockées dans un dossier pix/ dont le chemin absolu est /Users/oliviergimenez/Desktop/. 

Je voudrais évaluer les performances (vrais positifs, faux négatifs et faux positifs) du modèle entrainé par Gaspard Dussert sur un échantillon des photos du Jura annotées par Anna Chaine à reconnaître les espèces qui sont sur ces photos, et en particulier lynx, chamois et chevreuils. 

Ci-dessous on trouvera les différentes étapes du pipeline. C'est un mix de scripts R et Python. On applique une procédure en 2 étapes, détection puis classification. La même idée est appliquée par d'autres pour des projets (et avec des moyens) beaucoup plus ambitieux, voir par exemple [celui-ci](https://medium.com/microsoftazure/accelerating-biodiversity-surveys-with-azure-machine-learning-9be53f41e674). 

Le gros du boulot (en particulier l'entrainement d'un modèle de classification, cf. étape 4) a été fait par Gaspard Dussert en stage en 2019 avec Vincent Miele. Plus de détails [sur le site dédié du GdR EcoStat](https://ecostat.gitlab.io/imaginecology/). 

## Etape 1. Redimensionnement.

On redimensionne d'abord les images. Pour ce faire, on applique ces quelques lignes de code dans R. On utilise le package magical qui appelle l'excellent petit logiciel imagemagick. Les photos contenues dans le répertoire /Users/oliviergimenez/Desktop/pix sont redimensionnées en 1024x1024 dans le répertoire /Users/oliviergimenez/Desktop/pix_resized. Le nom de chaque photo est affublé d'un resized pour les différentier des photos originales. Le résultat est téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/pIanPETOyYIPwnN). 

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

### a. On traite une seule photo.

On prend par exemple '1.3 D (145)resized.JPG', et on lui met un cadre là où un objet est détecté. Taper dans le Terminal : 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_file /Users/oliviergimenez/Desktop/pix_resized/1.3\ D\ \(145\)resized.JPG
```

Le traitement prend quelques secondes. Un cadre a été ajouté sur la photo traitée, ainsi que la catégorie de l'objet détecté et un degré de confiance :

![detections](https://github.com/oliviergimenez/DLcamtrap/blob/master/1.3%20d%20(145)resized_detections.jpg)

### b. On traite toutes les photos du dossier pix_resized. 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_dir /Users/oliviergimenez/Desktop/pix_resized/
```

Les photos avec cadre sont ajoutées dans le même répertoire pix_resized, leur nom est juste modifié avec l'ajout de 'detections' pour signifier qu'elles ont été traitées. Les photos avec détections peuvent être récupérées [ici](https://mycore.core-cloud.net/index.php/s/nMCUzlbSxR6pho9). On remarque que pour la photo déjà traitée au a., un autre cadre a été ajouté, et la photo a été doublée. Si pour les animaux, le taux de succès est de 100%, pour les véhicules, il est de 0%, et pour les humains ce taux est élevé, mais pas de 100%. Pour les photos vides, pas de faux positifs, ie pas de cadre là où pas d'objets. Vu notre objectif, celui de travailler sur les interactions entre lynx, chamois, et chevreuils, le fait de détecter tous les animaux, et de ne pas mettre des cadre là où il n'y a pas d'animaux, nous semble prometteur, et ok pour continuer.

### c. On récupère l'info dans un fichier. 

On ne touche pas aux photos, et on crée plutôt un fichier json dans lequel on récupère les coordonnées des cadres, ainsi que les catégories des objets détectés. Cette étape est indispensable pour passer à la suite. 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector_batch.py /Users/oliviergimenez/Desktop/megadetector_v3.pb /Users/oliviergimenez/Desktop/pix_resized/ /Users/oliviergimenez/Desktop/box_pix.json
```

Attention, avant de lancer la commande au-dessus, transférer les 47 photos avec objets encadrés obtenus aux points a. et b. Dans un autre répertoire, par exemple pix_resized_detections, sinon l'étape c. portera sur les 93 photos.

On peut ouvrir le fichier json ainsi créé avec un éditeur de texte. Ce fichier est téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/vIIezUrWq7qNYFk). On peut voir des blocs, chaque bloc correspondant au traitement d'une image. Par exemple : 

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

On a le nom de l'image, la catégorie de l'objet détecté (0 = vide, 1 = animal, 2 = personne, 3 = groupe, 4 = véhicule), le degré de confiance (conf) ainsi que les caractéristiques de la boîte associée à l'objet (xmin, ymin, width, height) où l'origine de l'image est en haut à gauche à (xmin, ymin) (les coordonnées sont en coordonnées absolues). Il nous faudra la boîte sous la forme (ymin, xmin, ymax, xmax) ; si detection = (detection[0],detection[1],detection[2],detection[3])=(xmin, ymin, width, height) est le format json, alors la correspondance est xmin = detection[0], ymin = detection[1], xmax = detection[0] + detection[2] et ymax = detection[1] + detection[3]. On arrangera ce fichier à l'étape d'après pour en extraire l'information pertinente. 

## Etape 3. Métadonnées test. 

Avant de passer à la classification des objet détectés, il nous faut créer un fichier test.csv qui contient le nom du fichier photo, les coordonnées du cadre de l'objet détecté (données par MegaDetector à l'étape 2) et le nom de l'espèce détectée (tag manuel). Pour ce faire, il faut d'abord récupérer les tags manuels dans les métadonnées des photos, puis récupérer les coordonnées des boîtes créées à l'étape 2 de détection, assembler ces deux fichiers, puis on stocke le tout dans un fichier csv. 

```
# load package to manipulate data
library(tidyverse)

# load package to extract metadata from pix
library(exifr)

# load package to process json files
library(jsonlite)

#-- first, get manual tags

# where the resized pix are
pix_folder <- "/Users/oliviergimenez/Desktop/pix_resized/"

# list all files in the directory with pix
file_list <- list.files(pix_folder, recursive=TRUE, pattern = "*.JPG", full.names = TRUE)

# get metadata
manual_tags <- read_exif(file_list) %>%
  as_tibble() %>%
  select(FileName, Keywords) %>%
  unnest(Keywords) %>%
  filter(!Keywords %in% c('D','46.1','15.1','15.2')) # certains tags ne passent pas bien

# display
manual_tags

# 46 pix but 47 rows, why?
manual_tags %>% count(FileName, sort = TRUE)

# pix Cdy00008 (5)resized.JPG has 2 tags, both have to do with humans
manual_tags %>% filter(FileName == 'Cdy00008 (5)resized.JPG')

# for convenience, let's get rid of 'frequentation humaine' which appears only once
manual_tags %>% count(Keywords)
manual_tags <- manual_tags %>% filter(Keywords != 'frequentation humaine')

#-- second, get box coordinates

# where the json file is
json_folder <- "/Users/oliviergimenez/Desktop/"

# read in the json file
pixjson <- fromJSON(paste0(json_folder,'box_pix.json'), flatten = TRUE)

# what structure?
str(pixjson)

# names
names(pixjson)

# categories are animal, human and vehicules
pixjson$detection_categories

# get pix only
pix <- pixjson$images
names(pix)

# unlist detections and bbox
box_coord <- pix %>% 
  as_tibble() %>%
  unnest(detections, keep_empty = TRUE) %>%
  unnest_wider(bbox) %>%
  rename(xmin = '...1',
         ymin = '...2',
         width = '...3',
         height = '...4',
         name = file,
         max_det_conf = max_detection_conf,
         confidence = conf) %>%
  mutate(xmin = xmin,
         xmax = xmin + width,
         ymin = ymin,
         ymax = ymin + height,
         category = as.numeric(category)) %>%
  select(name, category, max_det_conf, confidence, xmin, xmax, ymin, ymax) %>%
  mutate(category = if_else(is.na(category), 0, category),
         confidence = if_else(is.na(confidence), 1, confidence))
  
# display
box_coord

# several pix get more than one box, let's take those with max confidence in detection
box_coord <- box_coord %>% 
  group_by(name) %>%
  arrange(desc(confidence)) %>% 
  slice(1) %>% 
  ungroup()

#-- third, join two files and build test.csv

box_coord %>% 
  rename(FileName = name) %>%
  mutate(FileName = str_remove(FileName, '/Users/oliviergimenez/Desktop/pix_resized/')) %>%
  left_join(manual_tags, by = 'FileName') %>%
  mutate(FileName = paste0('/Users/oliviergimenez/Desktop/pix_resized/',FileName)) %>%
  select(FileName, xmin, ymin, xmax, ymax, Keywords) %>%
  filter(!is.na(xmin)) %>% # select only pix with a box
  mutate(xmin = floor(1024 * xmin),
         xmax = floor(1024 * xmax),
         ymin = floor(1024 * ymin),
         ymax = floor(1024 * ymax)) %>%
  filter(! Keywords %in% c('cavalier','vehicule','humain','chien','vide','oiseaux')) %>%
  mutate(Keywords = fct_recode(Keywords, 'lièvre' = 'lievre')) %>%
  mutate(Keywords = fct_recode(Keywords, 'chat forestier' = 'chat')) %>%
  write_csv(paste0(json_folder,'test.csv'), 
            col_names = FALSE)
```

Le fichier crée peut être récupéré [là](https://mycore.core-cloud.net/index.php/s/5PYIlSqpzzcC5RX). A noter qu'on a supprimé de ce fichier toutes les photos dans lesquelles aucun objet n'a été détecté (pas de boîte) et celles qui ne correspondent pas à une catégorie entrainée. Cette étape devrait faire partie de l'évaluation des performances. 

## Etape 4. Classification. 

Pour classifier nos photos avec le modèle entrainé sur les photos du Jura taggées par Anna Chaine, on suit les étapes données [ici](https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet/). Le modèle déjà entrainé est téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/Prj6xeu0GqNWaXB), son petit nom est resnet50_csv_10.h5. 

Il nous faut le fichier [class.csv](https://mycore.core-cloud.net/index.php/s/gJkqIMK92ZyDFvK) qui contient les espèces sur lesquelles on a entrainé l'algorithme. On a déjà le fichier test.csv créé à l'étape précédente. 

Les étapes sont les suivantes pour la classification sont les suivantes : 

* On télécharge keras-retinanet [ici](https://github.com/fizyr/keras-retinanet).
* Aller dans le dossier keras-retinanet puis faire
```
pip install numpy --user 
```
et
```
pip install . --user
```

On peut vérifier le fichier des annotations qu'on a créé à l'étape précédente :
```
/Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/debug.py --annotations csv test.csv class.csv
```

Si on obtient ModuleNotFoundError: No module named 'keras_retinanet.utils.compute_overlap', on peut appliquer la commande suivante pour régler le problème dans le répertoire où se trouve setup.py :
```
cd keras-retinanet
python setup.py build_ext --inplace
```

On recommence : 
```
cd ..
/Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/debug.py --annotations csv test.csv class.csv
```

On fait la classification sur les photos qui ont déjà le cadre de la détection : 

```
/Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/evaluate.py --convert-model --save-path pix_pred/ --score-threshold 0.5 csv test.csv class.csv resnet50_csv_10.h5
```

On peut voir les photos classifiées dans le répertoire pix_pred/ avec le cadre en vert de l'étape détection par MegaDetector (qui est systématiquement trop bas) et celui en bleu de la classification par RetinaNet. Les photos peuvent être téléchargées [ici](https://mycore.core-cloud.net/index.php/s/gIkolFLoNuiT1lM). 

On peut faire le même exercice, mais en ne supprimant aucun photo, en ne gardant que le tag manuel, et en voyant ce que RetinaNet donne. Dans le script R précédent, modifier la dernière partie pour avoir :

```
box_coord %>% 
  rename(FileName = name) %>%
  mutate(FileName = str_remove(FileName, '/Users/oliviergimenez/Desktop/pix_resized/')) %>%
  left_join(manual_tags, by = 'FileName') %>%
  mutate(FileName = paste0('/Users/oliviergimenez/Desktop/pix_resized/',FileName)) %>%
  mutate(colxmin = NA,
         colymin = NA,
         colxmax = NA,
         colymax = NA,
         colkeywords = NA) %>%
  select(FileName, colxmin, colymin, colxmax, colymax, colkeywords) %>%
  write_csv(paste0(json_folder,'test2.csv'), 
            col_names = FALSE,
            na = '')
```

Le fichier test2.csv est [ici](https://mycore.core-cloud.net/index.php/s/XYEoMF0Puz6FMqV). Puis dans le Terminal, faire :
```
/Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/evaluate.py --convert-model --save-path pix_pred2/ --score-threshold 0.5 csv test2.csv class.csv resnet50_csv_10.h5
```

Les résultats sont téléchargeables [là](https://mycore.core-cloud.net/index.php/s/z2gcio7JgXGd7jt).

Avec qqs dizaines de photos, il est relativement facile d'évaluer les performances (étape suivante) de la classification. Maintenant si on a beaucoup de photos, on aimerait récupérer l'information brute. Pour afficher à l'écran (dans le Terminal) le nom de la photo, l'espèce détectée, la précision, et les coordonnées de la boîte, on utilise un script Python écrit par Vincent Miele, et téléchargeable via [ce lien](https://gitlab.com/ecostat/imaginecology/-/raw/master/projects/cameraTrapDetectionWithRetinanet/detect2txt.py)

## Etape 5. Evaluation. 

On évalue les performances avec script R. 

(5. Évaluer les performances TP, FN, FP avec script R postprocessML.R)

/Users/oliviergimenez/Desktop/DLcameratraps/keras-retinanet-master/keras_retinanet/bin/evaluate.py --convert-model --save-path test_pred/ --





 :

"J'ai ajouté un script Python décrit en bas de la page
https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet

Il s'agit de "detect2txt.py" :
-il te faut faire un "git pull"
-il te faut éditer le fichier et modifier les 3 paramètres (ceux qui
sont par défaut marchent si tu as suivi le tuto)
-et puis lancer "python3 detect2txt.py"" ; chez moi il faut ajouter le chemin absolu python3 /Users/oliviergimenez/Desktop/DLcameratraps/keras-retinanet-master/keras_retinanet/bin/detect2txt.py et avoir au préalable installer deux librairies manquantes, matplotlib et pandas

Gaspard fournit 2 scripts detect.py et detect2.py en pj. La différence entre les deux fichiers est dans la façon de calculer les faux négatifs. Le script detect.py calcule TP, FN et FP alors que detect2.py permet d'aller un peu plus dans le détail et de séparer les faux négatifs en FNvoid si l'animal n'est pas détecté et FN_false si l'animal a été détecté mais mal classifié. Ça fait suite à votre idée d'ajouter dans le background les classes avec peu de photos pour réduire le nombre de faux positifs. 

Note : le calcul de ces métriques d'erreur ne tiennent pas compte des erreurs faites à l'étape de la détection des objets avec MegaDetector. 
