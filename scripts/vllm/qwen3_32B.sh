#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4
vllm serve /media/public/models/huggingface/Qwen/Qwen3-32B/ --gpu-memory-utilization 0.95 --tensor-parallel-size 4