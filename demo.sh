#python demo.py \
#    --image input/1636351539931534.png input/1636351649229239.png \
#    --config configs/mg3-base.yaml \
#    --checkpoint logs/exp001/ckpts/checkpoint.pt \
#    --outdir output \
#    --resolution 448 \

python demo.py \
    --image input/1636351539931534.png input/1636351649229239.png \
    --depth input/1636351539931534_depth.npy input/1636351649229239_depth.npy \
    --config configs/mg3-base.yaml \
    --checkpoint logs/exp001/ckpts/checkpoint.pt \
    --outdir output \
    --resolution 448 \