# Piégeage photo et pipeline pour l'identification d'espèces

On va utilisé les méthodes de deep learning ou aprentissage profond pour faire l'identification automatique des espèces sur les images collectées via piégeages photographiques. Pour une introduction au deep learning (niveau lycée, d'après l'auteur), voir [ici](https://t.co/aAFS14fJuN?amp=1). Pour une introduction avec R au deep learning qui permet aussi de comprendre les réseaux de neurones, voir [là](http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Tensorflow_Keras_R.pdf). Je conseille [ça](https://agentmorris.github.io/camera-trap-ml-survey/) pour une veille sur deep learning et piégeage photographique. 

Passons à nos affaires. Comme exemple, j'ai assemblé un ensemble de 46 photos annotées à la main téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/ub5iTNSktszLvCv). Attention, bien s'assurer qu'il n'y a pas d'espace dans le nom des fichiers, et que les étiquettes (tags) ne comportent pas d'erreurs (sous Mac, on peut utiliser Photo pour modifier ces tags). L'information sur ce qui a été détecté dans chaque photo apparait dans les métadonnées des photos. Sous Mac, il suffit de faire un Cmd + I pour avoir cette info. Les photos sont stockées dans un dossier `pix/` dont le chemin absolu est `/Users/oliviergimenez/Desktop/`. 

On souhaite utiliser un modèle déjà existant, et évalué les performances de ce modèle à reconnaître les espèces qui sont sur les 46 photos de mon échantillon, et en particulier lynx, chamois et chevreuils. Ce modèle a été entrainé à classifier les espèces sur un échantillon des photos du Jura annotées par Anna Chaine. 

Pour évaluer ces performances, on va se concentrer sur : 
* les **vrais positifs ou TP** : le modèle prédit que l'espèce d'intérêt est présente sur la photo quand celle-ci est effectivement présente sur cette photo ; 
* les **faux positifs ou FP** : le modèle prédit la présence de l'espèce d'intérêt, mais celle-ci n'est en fait pas présente sur la photo ;
* les **faux négatifs ou FN** : le modèle prédit que l'espèce d'intérêt n'est pas présente sur la photo alors que celle-ci est bien présente ; on pourra séparer les FN en FN_void Fsi l'espèce n'a pas été détectée, et FN_false si l'espèce a été détectée mais mal classifiée. 

Ci-dessous on trouvera les différentes étapes du pipeline. C'est un mix de scripts R et Python. On applique une procédure en 2 étapes, détection puis classification. La même idée est appliquée par d'autres pour des projets (et avec des moyens) beaucoup plus ambitieux, voir par exemple [celui-ci](https://medium.com/microsoftazure/accelerating-biodiversity-surveys-with-azure-machine-learning-9be53f41e674). 

Le gros du boulot (en particulier l'entrainement du modèle) a été fait par Gaspard Dussert en stage en 2019 avec Vincent Miele. Plus de détails [sur le site dédié du GdR EcoStat](https://ecostat.gitlab.io/imaginecology/).

## Etape 0. Entrainement. 

Cette étape est donc déjà faite, ça tombe bien car les temps de calculs sont conséquents même avec des moyens performantes (GPU plutôt que CPU). On décrit brièvement les différentes étapes pour en arriver au modèle entrainé (voir [ce tuto](https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/cameraTrapDetectionWithRetinanet/) pour plus de détail sur l'entrainement d'un modèle). Au total, on a utilisé 8800 photos pour le training, divisées en 7992 pour l'entrainement/validation et 888 pour les tests. 

### a. Coup d'essai. 
On a estimé un modèle RetinaNet (voir la publication originale [ici](https://arxiv.org/abs/1708.02002) et une introduction [là](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/)) avec 9 classes jugées intéressantes et avec suffisamment d'images pour le training. Il s'agit de blaireaux, chamois, chat forestier, chevreuil, lièvre, lynx, sanglier, renard et cerf. Le tout a pris 24h sur avec GPU. Des tests montrent qu'on s'en sort très bien pour trouver ces 9 espèces quand elles sont présentes, mais sur des images qui contiennent des objets de classes qu'on a pas intégrées dans le modèle (e.g., humain, véhicules, vaches), le modèle détecte trop de faux positifs. 
### b. C'est mieux mais pas encore ça.
La solution envisagée a été d'ajouter des exemples négatifs dans le training, c'est à dire des images où il n'y a pas d'annotations (des vrais négatifs), avec des photos d'humains, de véhicules, de chiens et de vaches (mais sans les inclure en tant que classes du modèle, juste du background varié). A noter aussi qu'on n'a pas ajouté des exemples négatifs de toutes les classes : on n'a pas mis les fouines, martres, écureuil, oiseaux, lapins et toutes les autres classes avec peu de photos, en espérant que le modèle puisse s'en sortir sans apprendre une classe par animal de la planète. On obtient ainsi beaucoup moins de faux négatifs sur les classes qu'on a ajouté sans annotations. En revanche, il y a toujours des faux positifs sur les espèces qu'on n'a pas mis dans les exemples négatifs et qui ressemblent à des espèces du modèle : lapin avec lièvre, chat domestique avec chat forestier, fouine ou martre avec renard et blaireaux. Avec des espèces qui ne se ressemblent pas comme oiseaux et écureuils, il n'y a pas de faux positifs mais il y aussi peu d'images pour le confirmer. 
### c. Voilàààààà. 
La dernière étape a donc été d'ajouter dans le background les classes avec peu de photos, entre autres fouine, martre, vache, et léporidés. Ce faisant, on a grandement diminué le nombre de faux positifs du modèle tout en conservant un nombre de faux négatifs semblable. 

## Etape 1. Redimensionnement.

Passons maintenant au coeur de l'exercice, la classification automatique des 46 photos. On redimensionne d'abord les images. Pour ce faire, on applique ces quelques lignes de code dans `R`. On utilise le package magical qui appelle l'excellent petit logiciel `imagemagick`. Les photos contenues dans le répertoire `/Users/oliviergimenez/Desktop/pix` sont redimensionnées en 1024x1024 dans le répertoire `/Users/oliviergimenez/Desktop/pix_resized`. Les chemins sont à modifier selon les goûts. Le nom de chaque photo est affublé d'un resized pour les différentier des photos originales. Le résultat est téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/pIanPETOyYIPwnN). 

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

Si l'on inspecte les photos redimensionnées, on s'aperçoit que leur poids a été divisé par 5-10 en gros, d'1-1.5Mo en moyenne, à 200Ko en moyenne. 

## Etape 2. Détection. 

On passe maintenant à la détection des objets dans les photos. Pour ce faire, on utilise [MegaDetector](https://github.com/microsoft/CameraTraps#overview) pour se faciliter la vie. Cet algorithme va détecter les objets sur les photos et leur associer un cadre, une boîte. 

Pour ce faire, il faut d'abord télécharger depuis [CameraTraps](https://github.com/microsoft/CameraTraps) le fichier zippé, le dézipper, puis changer le nom ddu répertoire en `CameraTraps`. Puis, depuis un Terminal, se mettre dans le répertoire `CameraTraps/` et suivre [les instructions d'installation](https://github.com/microsoft/CameraTraps#initial-setup). En gros cela consiste à taper les deux lignes de commande 
```conda env create --file environment.yml``` 
et 
```conda env create --file environment-detector.yml```. 
Attention, si on n'a pas de GPU, il faut modifier le fichier dans le fichier environment-detector.yml en commentant la ligne 
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

On prend par exemple '1.3D(145)resized.JPG', et on lui met un cadre là où un objet est détecté. Taper dans le Terminal : 

```
python /Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py /Users/oliviergimenez/Desktop/megadetector_v3.pb --image_file /Users/oliviergimenez/Desktop/pix_resized/1.3D\(145\)resized.JPG
```

Le traitement prend quelques secondes. Un cadre a été ajouté sur la photo traitée, ainsi que la catégorie de l'objet détecté et un degré de confiance :

![detections](https://github.com/oliviergimenez/DLcamtrap/blob/master/1.3%20d%20(145)resized_detections.jpg)

J'ai eu une erreur quand j'ai lancé la ligne de commande ci-dessus qui disait : 
```
Traceback (most recent call last):
  File "/Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py", line 54, in <module>
    from ct_utils import truncate_float
ModuleNotFoundError: No module named 'ct_utils'
```

J'ai copié et collé le script `ct_utils` de `/Users/oliviergimenez/Desktop/CameraTraps` vers `/Users/oliviergimenez/Desktop/CameraTraps/detection`. J'ai alors encore eu une erreur avec le message : 
```
Traceback (most recent call last):
  File "/Users/oliviergimenez/Desktop/CameraTraps/detection/run_tf_detector.py", line 55, in <module>
    import visualization.visualization_utils as viz_utils
ModuleNotFoundError: No module named 'visualization'
```

J'ai installé `freetype` et `pkg-config` via un `brew install`. 

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

Avec qqs dizaines de photos, il est relativement facile d'évaluer les performances (étape suivante) de la classification. Maintenant si on a beaucoup de photos, on aimerait récupérer l'information brute. Pour afficher à l'écran (dans le Terminal) le nom de la photo, l'espèce détectée, la précision, et les coordonnées de la boîte, on utilise un script Python écrit par Vincent Miele, et téléchargeable via [ce lien](https://gitlab.com/ecostat/imaginecology/-/raw/master/projects/cameraTrapDetectionWithRetinanet/detect2txt.py?inline=false). J'ai placé ce script dans le répertoire où se trouve evaluate.py à savoir keras-retinanet/keras_retinanet/bin/. Pour utiliser ce script, il faut l'éditer avec un éditeur texte par exemple, et modifier les 3 lignes suivantes :
```
model_path = "retinanet_tutorial_best_weights.h5" # model file (must be based on RESNET50)
test_dir = "test" # test directory with the target images
classfile = "class.csv" # list of class labels
```
pour préciser le nom de notre modèle entrainé, le répertoire avec les images à classifier et le fichier avec les catégories, soit :
```
model_path = "resnet50_csv_10.h5" # model file (must be based on RESNET50)
test_dir = "pix_resized" # test directory with the target images
classfile = "class.csv" # list of class labels
```

Alors, dans le Terminal, on utilise une série de commandes similaires à celle utilisées précédemment : 
```
python /Users/oliviergimenez/Desktop/keras-retinanet-master/keras_retinanet/bin/detect2txt.py
```

Les résultats s'affichent à l'écran, avec le nom de fichier, l'espèce identifiée, la confiance, et les coordonnées du cadre. Plus pratique, on peut utiliser la redirection dans un fichier texte :
```
python /Users/oliviergimenez/Desktop/keras-retinanet-master/keras_retinanet/bin/detect2txt.py >> classif_pix.txt
```
avec le fichier ainsi créé téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/nQZOyFG3bxVAzpW). A noter que dans le nom des fichiers, il y a des espaces, parfois plusieurs par fichier, je les ai enlevés à la main ici. J'aurais du le faire au tout début, sous R par exemple, en utilisant str_remove_all().  

## Etape 5. Evaluation. 

Pour évaluer la qualité de la classification, on compare les tags manuels au catégories prédites par l'algorithme. On fait ça sous R : 

```
# load suite of packages to manage/visualise data
library(tidyverse)

# remove blanks in pix names of manual tags
manual_tags <- manual_tags %>%
  mutate(FileName = str_remove_all(FileName,' '))
print(manual_tags, n = Inf)

# read in classifications
classif <- read_tsv('/Users/oliviergimenez/Desktop/classif_pix.txt', col_names = FALSE) %>%
  mutate(X1 = str_replace(X1, 'chat forestier', 'chat_forestier')) %>%
  filter(!str_detect(X1, 'tracking')) %>%
  separate(X1, 
           into = c("name","species","confidence","x","y","z","t"), 
           sep = '\\s') %>%
  mutate(x = str_remove(x,'\\['),
         x = str_remove(x,','),
         y = str_remove(y,','),
         z = str_remove(z,','),
         t = str_remove(t,'\\]'),
         t = str_remove(t,','),
         x = as.numeric(x),
         y = as.numeric(y),
         z = as.numeric(z),
         t = as.numeric(t)) %>%
  rename(FileName = name) %>%
  select(FileName, species, confidence, x, y , z, t) %>%
  mutate(FileName = str_remove(FileName, 'pix_resized/'))
print(classif, n = Inf)

# comparison
manual_tags %>% left_join(classif) %>% rename(ground_truth = Keywords, prediction = species) %>% print(n=Inf)


Joining, by = "FileName"
# A tibble: 51 x 8
   FileName                                                                       ground_truth   prediction     confidence     x     y     z     t
   <chr>                                                                          <chr>          <chr>          <chr>      <dbl> <dbl> <dbl> <dbl>
 1 1.3D(145)resized.JPG                                                           chamois        chamois        0.99999046    96   190   833   638
 2 1.3D(146)resized.JPG                                                           chamois        chamois        0.99999255     0   343   668   744
 3 1.3D(189)resized.JPG                                                           chamois        chamois        0.99998736   200   122   849   658
 4 1.3D(208)!!ATTENTION2017AULIEUDE2016!!resized.JPG                              cerf           cerf           0.9998793      6     0   973   576
 5 Cdy00002(2)resized.JPG                                                         cavalier       NA             NA            NA    NA    NA    NA
 6 Cdy00004(4)resized.JPG                                                         renard         renard         0.9788616    524   354   726   474
 7 Cdy00004(4)resized.JPG                                                         renard         sangliers      0.51794666   526   353   720   463
 8 Cdy00005(4)resized.JPG                                                         chevreuil      chevreuil      0.9999309     64   263   574   584
 9 Cdy00005(5)resized.JPG                                                         vehicule       NA             NA            NA    NA    NA    NA
10 Cdy00007(5)resized.JPG                                                         chat forestier chat_forestier 0.99999195   405   325   636   488
11 Cdy00008(5)resized.JPG                                                         vehicule       NA             NA            NA    NA    NA    NA
12 Cdy00008resized.JPG                                                            chevreuil      chamois        0.9990773    507   206   613   364
13 Cdy00008resized.JPG                                                            chevreuil      chevreuil      0.81188613   507   206   613   364
14 Cdy00011(2)resized.JPG                                                         chevreuil      chamois        0.9996481    360   265   472   491
15 Cdy00011(7)resized.JPG                                                         vehicule       NA             NA            NA    NA    NA    NA
16 Cdy00013(5)resized.JPG                                                         vehicule       NA             NA            NA    NA    NA    NA
17 Cdy00020resized.JPG                                                            renard         renard         0.9999999    665   396  1023   556
18 FDC01_point_15-4a79b79d.2_corlier_montlier_2018_7_23_flanc_droit(3)resized.JPG lynx           lynx           0.99994326   704   428   923   564
19 FDC01_point_36.2_evosges_le_col_2017_09_10_flanc_droitresized.JPG              lynx           lynx           0.99999857     0   318   310   565
20 FDC01_point_36.2_evosges_le_col_2017_09_29_cuissegaucheresized.JPG             lynx           lynx           0.99994063     0   330   336   652
21 FDC01_point_36.2_evosges_le_col_2017_10_10_flanc_droitresized.JPG              lynx           lynx           0.99994177   264   346   957   755
22 I__00001(7)resized.JPG                                                         humain         NA             NA            NA    NA    NA    NA
23 I__00002(7)resized.JPG                                                         vide           NA             NA            NA    NA    NA    NA
24 I__00003(8)resized.JPG                                                         vide           NA             NA            NA    NA    NA    NA
25 I__00007(8)resized.JPG                                                         renard         renard         0.9997709      1   392   527   692
26 I__00009(8)resized.JPG                                                         sangliers      NA             NA            NA    NA    NA    NA
27 I__00010(8)resized.JPG                                                         sangliers      sangliers      0.8085136     11   244  1003   762
28 I__00012(6)resized.JPG                                                         vide           NA             NA            NA    NA    NA    NA
29 I__00012resized.JPG                                                            sangliers      sangliers      0.9999937    422     1  1017   477
30 I__00015(10)resized.JPG                                                        chien          NA             NA            NA    NA    NA    NA
31 I__00015(3)-8907af1cresized.JPG                                                vide           NA             NA            NA    NA    NA    NA
32 I__00015resized.JPG                                                            chevreuil      chevreuil      0.9984863    675   184   934   448
33 I__00016(6)resized.JPG                                                         vide           NA             NA            NA    NA    NA    NA
34 I__00016resized.JPG                                                            vide           NA             NA            NA    NA    NA    NA
35 I__00017(7)resized.JPG                                                         humain         NA             NA            NA    NA    NA    NA
36 I__00020(6)resized.JPG                                                         chevreuil      chevreuil      0.9999998    208   128   863   717
37 I__00021resized.JPG                                                            blaireaux      blaireaux      0.9999969    371   339   580   434
38 I__00022(7)resized.JPG                                                         chien          NA             NA            NA    NA    NA    NA
39 I__00023(7)resized.JPG                                                         chat           chat_forestier 0.77948236   739   497  1022   756
40 I__00024(8)resized.JPG                                                         chat           renard         0.96120846   546   387  1022   756
41 I__00024(8)resized.JPG                                                         chat           chamois        0.7607192    555   382  1022   755
42 I__00025(10)resized.JPG                                                        chevreuil      chevreuil      0.9999732    282   217  1020   759
43 I__00026(11)resized.JPG                                                        sangliers      sangliers      0.9359536    473   303  1022   762
44 I__00028resized.JPG                                                            lievre         lièvre         0.99646026   734   301  1022   624
45 I__00028resized.JPG                                                            lievre         lynx           0.88849044   734   301  1022   624
46 I__00033(6)resized.JPG                                                         oiseaux        NA             NA            NA    NA    NA    NA
47 I__00033(9)resized.JPG                                                         humain         NA             NA            NA    NA    NA    NA
48 I__00049(6)resized.JPG                                                         chat forestier lièvre         0.9782964    145   355   508   689
49 I__00049(6)resized.JPG                                                         chat forestier chat_forestier 0.56414235   143   353   509   690
50 I__00051(10)resized.JPG                                                        lievre         lièvre         0.99998236   281   438   664   725
51 I__00060(10)resized.JPG                                                        humain         NA             NA            NA    NA    NA    NA
```

Sur notre exemple, avec peu de photos, on peut regarder ce qui se passe. Pour les 4 premières photos, la prédiction coincide avec la vérité avec une grande confiance. Sur la photo 5, on a un cavalier, et cette photo a été écartée à l'étape de la détection puisque les coordonnées sont manquantes. Idem pour toutes les photos avec véhicules, humain, chien, oiseaux ou encore les photos vides. Très bien. La photo 12-13 montre qu'on confond chevreuil et chamois (c'est bien le même cadre, il suffit de regarder les coordonnées). Le message est encore plus clair avec la photo 14. Les photos 18 à 21 montrent que le lynx est bien classifié. La photo 40-41 montre qu'on confond le chat avec renard ou chamois, mais plutôt renard à en jugerpar le degré de confiance. Sur la photo 44-45, on hésite pour le lièvre entre lièvre et lynx, mais quand on regarde le degré de confiance, on penche pour le lièvre. Dans la photo 48-49 on hésite aussi pour le chat forestier entre lièvre et chat forestier, et on se plante si on regarde le degré de confiance puisqu'on prendrait le lièvre. 

On peut formaliser ces calculs d'erreur en distinguant :
* les faux négatifs quand une espèce présente sur la photo n'est pas détectée ; on peut redécouper en Void si animal pas détecté et False si animal détecté mais mal classifié ; 
* les vrais positifs quand une espèce présente sur la photo est détecté et bien classifiée ;
* les faux positifs quand une espèce n'est pas sur une photo mais y est détectée.

Pas facile de faire ces calculs à la main quand on a beaucoup de photos. Heureusement, Gaspard Dussert fournit un script Python detect.py téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/62ORvHlZEpNdA5y) qui permet de faire ces calculs. Je mets ce script dans /Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/ et il ne manque plus qu'à modifier deux lignes du script. 

Au début, modifier la ligne 
```
model_path = "snapshots_all/resnet50_csv_10.h5"
```
en
```
model_path = "/Users/oliviergimenez/Desktop/resnet50_csv_10.h5"
```
pour dire où se trouve le modèle entrainé (chemin absolu, comme d'habitude).

A la fin du script, modifier la ligne
```
comp_exif("/beegfs/data/gdussert/projects/olivier_pipeline/all_classes/test/")
```
en 
```
comp_exif("/Users/oliviergimenez/Desktop/pix_resized/")
```
pour dire où sont les photos à classifier. 

Enfin, dans le Terminal, faire :
```
python /Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/detect.py
```

Sur le Terminal, on a aussi le détail comme suit : 

```
          species  TP  FP  FN_false  FN_void
0       blaireaux   0   1         0        0
1         chamois   3   2         0        0
2  chat forestier   1   1         1        0
3       chevreuil   2   2         0        0
4          lièvre   0   3         0        0
5            lynx   4   0         0        0
6          renard   1   3         0        0
7       sangliers   3   0         0        1
8            cerf   1   0         0        0

Images source de FP par classe :
46.1 5
15.1 3
chat 2
lievre 1
chat forestier 1

Nombre total d'image par classe:
vide 5
46.1 5
lynx 4
sangliers 4
15.1 4
chevreuil 2
vehicule 4
chat 2
cerf 1
chat forestier 2
lievre 1
chien 2
humain 4
chamois 3
oiseaux 1
renard 1
cavalier 1
```

On peut, comme précédemment, mettre tout ça dans un fichier texte via une redirection :
```
python /Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/detect.py >> perf.txt
```

On peut jeter un coup d'oeil au résultat [là](https://mycore.core-cloud.net/index.php/s/GnJwuAnI2NUyin6).

Si l'on n'est pas forcément intéressé par le détail dans les faux négatifs, on peut utiliser un autre script Python, detect2.py, téléchargeable [ici](https://mycore.core-cloud.net/index.php/s/uAWT3lxQetkIH68), et qui permet de mettre sur les photos le cadre avec l'espèce vérité et le cadre avec l'espèce prédite, et de créer trois nouveaux répertoires qui contiennent les photos classées en TP, FN et FP, permettant ainsi d'aller regarder en détail les situations qui génèrent les faux négatifs et faux positifs. Ces répertoires se trouvent dans le répertoire qui contient les photos analysées, dans pix_resized/ ici.

Avant de lancer le script, il faut modifier les deux mêmes lignes que pour le script detect.py. Puis on le lance via :
```
python /Users/oliviergimenez/Desktop/keras-retinanet/keras_retinanet/bin/detect2.py >> perf2.txt
```

On peut jeter un coup d'oeil à perf2.txt : 
```
          species  TP  FP  FN
0       blaireaux   0   1   0
1         chamois   3   2   0
2  chat forestier   1   1   1
3       chevreuil   2   2   0
4          lièvre   0   3   0
5            lynx   4   0   0
6          renard   1   3   0
7       sangliers   4   0   0
8            cerf   1   0   0

Images source de FP par classe :
46.1 5
15.1 3
chat 2
lievre 1
chat forestier 1

Nombre total d'image par classe:
vide 5
46.1 5
lynx 4
sangliers 4
15.1 4
chevreuil 2
vehicule 4
chat 2
cerf 1
chat forestier 2
lievre 1
chien 2
humain 4
chamois 3
oiseaux 1
renard 1
cavalier 1
```

Au passage, ces scripts detect.py et detect2.py font en un coup les étapes 2, 3 et 4. 

## La suite ? 

* Vérifier que l'algo déjà entrainé n'est pas à côté de la plaque dans l'Ain en comparant les tags manuels entrés par les collègues de l'OFB à la classification prédite de l'algo. Si on n'est pas trop dans les choux, alors fair ele tagging automatique pour toutes les photos de l'Ain pour que Maëlis puisse se servir de ces données pour le stage. 

* A moyen terme, refaire l'entrainement du modèle avec toutes les photos du Jura annotées par Anna (?) et toutes les photos de l'Ain annotées par les collègues de l'OFB. Plus on a de photos, mieux c'est. Ce modèle pourra alors être utilisé pour tagger les photos restantes dans l'Ain, et pour tagger à l'avenir les photos collectées dans le cadre du PPP Lynx. 
