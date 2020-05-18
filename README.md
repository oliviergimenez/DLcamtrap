# DLcamtrap
My personal pipeline to species identification on camera trap pix using deep learning, detection/classification with MegaDetector and RetinaNet

1. Appliquer script resize.R pour mettre les pix aux bonnes dim

2. Appliquer MegaDetector pour récupérer les coordonnées des boîtes qui montrent les objets détectés (voir ci-dessous pour les commandes) ; https://gitlab.com/ecostat/imaginecology/-/tree/master/projects/detectionWithMegaDetector/

3. Extraire du fichier json créé à l'étape précédente les infos nécessaires aux prédictions en utilisant le script extract_from_json.R

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
