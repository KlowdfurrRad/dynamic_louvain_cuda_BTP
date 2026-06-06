nvcc -rdc=true -arch=sm_60 cuda_static_louvain.cu -o static_louvain
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain_nodebased.cu -o dynamic_louvain_nodebased
nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain
