mkdir face_detector/weights
mkdir mask_classifier/weights

gdrive download 1JhXo_NpsoY0-GELus0hZRZMwO-JO_ZfZ
gdrive download 1E-9kDwApGM_dA7lcbPZ32AYZsYSCpkGj

mv Resnet50_Final.pth face_detector/weights
mv resnet50.h5 mask_classifier/weights

mkdir data
gdrive download 1FttRaUr7sJZM5brXFuNo9cKCB2LNgdHv
unzip -qq train.zip -d data
rm train.zip
