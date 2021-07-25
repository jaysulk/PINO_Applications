#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 run_pino3d.py \
--config_path configs/transfer/Re300to350-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=2 python3 run_pino3d.py \
--config_path configs/transfer/Re350to350-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=2 python3 run_pino3d.py \
--config_path configs/transfer/Re400to350-1s.yaml \
--start 0 \
--stop 40 \
--log;
