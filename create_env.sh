conda create -y -n imaging_sim python=3.9
conda activate imaging_sim
conda install -y -c pytorch -c nvidia -c conda-forge pytorch=2.1.2 torchvision pytorch-cuda=12.1 matplotlib=3.8.0 numpy=1.26.4 scipy=1.11.3 tqdm

conda install -c pytorch -c nvidia -c conda-forge pytorch=2.1.2 torchvision pytorch-cuda=12.1 matplotlib=3.8.0 numpy=1.26.4 scipy=1.11.3 tqdm opencv