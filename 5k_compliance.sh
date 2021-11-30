python 5k_compliance.py \
        --weights crowdhuman_yolov5m.pt \
        --source data/public_test/images\
        --person \
        -i data/public_test/images\
        --load-ckpt ckpt/res50.pth \
        --backbone resnet50 \
        --save-txt