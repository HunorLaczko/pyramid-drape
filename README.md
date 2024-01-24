# pyramid-drape

### A Generative Multi-Resolution Pyramid and Normal-Conditioning 3D Cloth Draping (WACV24)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> 
<a href="https://hunorlaczko.github.io/projects/pyramid-drape/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2311.02700-b31b1b.svg)](https://arxiv.org/abs/2311.02700
)

# Instructions

## Data preparation

### Dataset 

We use the [CLOTH3D](https://chalearnlap.cvc.uab.cat/dataset/38/description/) dataset. You can find a description of it on the link provided. For direct downloading the test set you can use this [link](https://158.109.8.102/CLOTH3D/Test/test_t1.tar.gz). (you might receive a https/ssl warning, you can ignore it and continue)

As described in the paper we register each garment to the SMPL body fo unify their topology. This registration information can be found at this [link](https://uab-my.sharepoint.com/:f:/g/personal/1648039_uab_cat/EpUP9kD47SVJoyXFbfKtOw8B5-4IUrKAEHVp_wm6lxihPA?e=FhRYVH).

### Checkpoints
We provide two checkpoints for the final model, one for the non-skirt garments and one for the skirt-like garments. Please find them at the same [link](https://uab-my.sharepoint.com/:f:/g/personal/1648039_uab_cat/EpUP9kD47SVJoyXFbfKtOw8B5-4IUrKAEHVp_wm6lxihPA?e=FhRYVH). We provide two additional checkpoints that were further finetuned to improve sampling performance, in eachange for slightly higher reconstruction error. 

### SMPL

We include a modified version of the SMPL which has a higher resolution and has the hands, feet and head removed. For the original version and licensing information please hheck the original [https://smpl.is.tue.mpg.de/](webpage). 

## Running the code

Please note that the preprocessing of the data might require a longer time, and extra storage space. 

1. Clone this repository.
2. Move the downloaded checkpoints to a `checkpoints` folder in the root of the repository.
3. (Optional) Update `docker/Dockerfile` with the user information you want to use inside the docker container. (The default is root/root)
4. Update `docker/docker-compose.yml` with the folder mappings based on where you downloaded the dataset and where you want the preprocessed data to be stored as well as the outputs. 
5. Build and start the docker container using docker compose:
    ```
    docker compose build
    docker compose up -d
    ```
6. Connect interactive terminal to the container:
   ```
   docker exec -it pyramid-drape /bin/bash
   ```
7. Run preprocessing (update default command line argumens if necessary, like gpu id):
   ```
   source /opt/conda/bin/activate base
   python data/preprocessing.py
   python data/convert_to_uv.py
   python data/downscale.py
   ```
   7.1. (Optional) All preprocessing scripts support `--start` and `--end` arguments, so they can be run on a subset of the dataset only. As such, the scripts can be run in parallel on different subsets, significantly speeding up the processes. 
8. Run inference:
   ```
   source /opt/conda/bin/activate base
   python scripts/generate_results.py --config configs.pyr
   python scripts/generate_results.py --config configs.pyr_skirt
   ```
   This generates the output mesh coordinates and saves them as .npz files per frame.
9. Calculate error:
   ```
   python scripts/calculate_error.py --config configs.pyr
   python scripts/calculate_error.py --config configs.pyr_skirt
   ```
   This calculates the euclidean error for each frame and saves them to a .csv file

   Get final error:
   ```
   python scripts/get_error.py --config configs.pyr --config_skirt configs.pyr_skirt
   ```
   This merges the results of the skirt and non-skirt results to receive the final error. 

If you have any question, feel free to contact me or open a github issue.

## Citation
If you find our work useful in your research, please consider citing our paper:
```
@InProceedings{Laczko_2024_WACV,
    author    = {Laczk\'o, Hunor and Madadi, Meysam and Escalera, Sergio and Gonzalez, Jordi},
    title     = {A Generative Multi-Resolution Pyramid and Normal-Conditioning 3D Cloth Draping},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {8709-8718}
}
```