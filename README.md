# PhotoScene

The official implementation of **PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes**.<br/>
[Yu-Ying Yeh](http://yuyingyeh.github.io), [Zhengqin Li](https://sites.google.com/a/eng.ucsd.edu/zhengqinli), [Yannick Hold-Geoffroy](https://yannickhold.com), [Rui Zhu](https://jerrypiglet.github.io), [Zexiang Xu](https://cseweb.ucsd.edu/~zex014), [Miloš Hašan](http://www.miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)<br>
IEEE / CVF Computer Vision and Pattern Recognition Conference  (CVPR), 2022 <br>

[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yeh_PhotoScene_Photorealistic_Material_and_Lighting_Transfer_for_Indoor_Scenes_CVPR_2022_paper.pdf)] [[Project](https://yuyingyeh.github.io/projects/photoscene.html)]

![](https://github.com/yuyingyeh/yuyingyeh.github.io/blob/master/projects/photoscene/teaser.png)

## Prerequisite
1. Clone this repository including submodules
```
git clone --recursive https://github.com/ViLab-UCSD/PhotoScene.git
cd PhotoScene
```

2. Install [OptiX 5.1.1](https://developer.nvidia.com/designworks/optix/downloads/5.1.1/linux64) under `third_party/OptiX`. We assume OptiX install directory is `third_party/OptiX/NVIDIA-OptiX-SDK-5.1.1-linux64`. Note that the version should be 5.1.1 upon which our renderer is built.

3. Download [Total3D](https://github.com/yinyunie/Total3DUnderstanding), [MaskFormer](https://github.com/facebookresearch/MaskFormer), [InvRenderNet](https://github.com/lzqsd/InverseRenderingOfIndoorScene) pretrained model by running the following script.
```
bash scripts/download_model.sh
```

4. Set up Substance Designer
- Install [Substance Designer](https://www.adobe.com/products/substance3d-designer.html) with Linux (.rpm) version 2021.1.2

Get free license [here](https://store.substance3d.com/students-teachers) if you're a student or teacher. Download Substance Designer version 11.1.2 (437M) (`LICENSES`->`Substance Designer All builds`->`Linux (.rpm) version 2021.1.2`) and run the following to install. The default installation path will be `/opt/Allegorithmic/Substance_Designer`.
```
sudo apt-get update
sudo apt-get install alien
sudo alien -i Substance_Designer-11.1.2-4593-linux-x64-standard.rpm
```
- Note: It is a requirement for [DiffMat](https://github.com/mit-gfx/diffmat)

## Requirement
1. Install [Docker](https://www.docker.com)

Our framework consists of multiple dependencies. We strongly recommend using our provided docker image to run this repository. The provided script assumes running docker in [rootless mode](https://docs.docker.com/engine/security/rootless/) with a single GPU. Might need to modify `/etc/nvidia-container-runtime/config.toml` for the entry under [nvidia-container-cli] to `no-cgroups = true` to disable the use of cgroups by the NVIDIA container runtime. After installation, run the following to pull our image and launch the environment.
```
docker pull yyeh/photoscene:v1
bash scripts/run_docker_interactive.sh
```

2. Build [OptixRenderer](https://github.com/lzqsd/OptixRenderer)

Please make sure OptiX 5.1.1 is installed under `third_party/OptiX` before running the following script. Note that from this step all the commands must be run inside the docker environment.
```
bash scripts/build_renderer.sh
```

## Run PhotoScene
Now we can run the full PhotoScene framework on a scene specified by `$yamlFile`.
```
bash run_photoscene.sh $yamlFile
```
For example, `$yamlFile` is set default as `configs/total3d/Total_246.yaml`.
Alternatively, you can run each single step sequentially:
```
# Initialization and Alignment
python3 photoscene/preprocess.py --config $yamlFile

# Graph Selection
python3 photoscene/selectGraphFromCls.py --config $yamlFile

# First Round Material Optimization
python3 photoscene/optimizeMaterial.py --config $yamlFile --mode first

# Lighting Optimization
python3 photoscene/optimizeLight.py --config $yamlFile

# Second Round Material Optimization
python3 photoscene/optimizeMaterial.py --config $yamlFile --mode second

# Render Final PhotoScene Result
python3 photoscene/renderPhotoScene.py --config $yamlFile
```
## Additional Input Data
You can first use provided Total3D examples to run the entire framework. If you want to try more scenes, please download Total3D preprocessed data by running the following script or following [Total3D](https://github.com/yinyunie/Total3DUnderstanding#data-preparation) to download preprocessed SUN RGBD data (12G) to `third_party/Total3D/data/sunrgbd/sunrgbd_train_test_data` and copy the scenes (<scene_id>.pkl) to `data/total3d/inputs`. 
```
bash scripts/download_total3d_data.sh
```

## Citation

Please cite our paper if you find that our method is helpful!

```BibTeX
@InProceedings{Yeh_2022_CVPR,
    author    = {Yeh, Yu-Ying and Li, Zhengqin and Hold-Geoffroy, Yannick and Zhu, Rui and Xu, Zexiang and Ha\v{s}an, Milo\v{s} and Sunkavalli, Kalyan and Chandraker, Manmohan},
    title     = {PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18562-18571}
}
```
