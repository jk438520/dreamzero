# export LOAD_TRT_ENGINE=checkpoints/DreamZero-DROID/tensorrt/wan/WanModel_nvfp4.trt 
export DYNAMIC_CACHE_SCHEDULE=true 
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 8123 --enable-dit-cache --model-path checkpoints/DreamZero-DROID