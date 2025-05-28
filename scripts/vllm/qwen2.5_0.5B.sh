#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-0}
vllm serve /media/public/models/huggingface/Qwen/Qwen2.5-0.5B-Instruct