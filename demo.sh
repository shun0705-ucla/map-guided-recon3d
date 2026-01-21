python demo.py \
    --image input/1st.png input/3rd.png \
    --depth input/1st_depth_xvader.npy input/3rd_depth_xvader.npy \
    --config configs/mg3-base.yaml \
    --checkpoint checkpoints/mg3/mg3_base_init.pth \
    --outdir output \
    --resolution 448 \