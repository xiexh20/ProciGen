# ProciGen
[Project Page](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/) | [ProciGen Dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.2VUEUS ) | [Model trained in ProciGen](https://github.com/xiexh20/HDM) | [Paper](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/paper-lowreso.pdf)

ProciGen is a synthetic human-object interaction dataset with **1M+ images** of human interacting with **21k+ different objects**.
We achieve this by procedurally combining human, object and interaction datasets together. This repo provides code to access
and generate ProciGen dataset.  
<p align="left">
<img src="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/gif_procigen.gif" alt="teaser" width="80%"/>
</p>

[//]: # (ProciGen stands for **Proc**edural **i**nteraction **Gen**eration. We procedurally combines human, object and interaction datasets together, )

[//]: # (which allows us to generate an interaction dataset of 1M+ images with 21k+ different objects. We call this dataset ProciGen.  )

## Contents
1. [Dependencies](#dependencies)
2. [Dataset Structure](#dataset-structure)
3. [Synthesize ProciGen](#synthesize-procigen)
4. [License](#license)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

### TODO List
- [x] Dataset download links.
- [x] Dataset structure.
- [ ] Autoencoder to compute dense correspondence. 
- [ ] Contact transfer and optimization.
- [ ] Render new human object interaction. 

## Dependencies
The code is tested on `python 3.8.16, Debian 10`. The runtime environment is the same as HDM, please follow the instructions [here](https://github.com/xiexh20/HDM#dependencies).

## Dataset Structure

**Download links:** dataset can be downloaded from [edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.2VUEUS). By downloading the dataset, you agree the [LICENSE](./LICENSE).

We follow the similar structure as the [behave dataset](https://github.com/xiexh20/behave-dataset?tab=readme-ov-file#dataset-structure). 
Namely, each sequence is organized as follows:
```
|--ROOT
|----sequence_name
|------info.json  # a file storing the calibration information for the sequence
|------<frame_name>     # one frame folder
|--------k*.color.jpg               # color images of the frame
|--------k*.person_mask.png         # human masks
|--------k*.obj_rend_mask.png       # object masks
|--------k*.obj_rend_full.png       # object masks without occlusion
|--------k*.depth.png               # [optional] depth images  
|--------k*.grid_df_res128_b0.5.npz # [optional] precomputed occupancy for human object segmentation
|--------person
|----------fit01                    # registered SMPL-H mesh and parameters
|--------<object_name>
|----------fit01                    # object registrations

```
### Human Model
We store the SMPL-H parameters and corresponding mesh inside each `person/fit01` folder. We use the MANO_V1.2 pkl model with 10 shape parameters and no PCA compression for the hand poses. If you would like to use other body models e.g. SMPL or SMPL-X, please refer to [this repo](https://github.com/vchoutas/smplx/tree/master/transfer_model) for conversions between different body models. 


### Object Model
We save the simplified object mesh in file `<object_name>/fit01/<object_name>_fit.ply`. 
The corresponding object pose and other information are saved in `<object_name>/fit01/<object_name>_fit.pkl`. 
Inside the pickle file, you can find the fields `synset_id` and `ins_name`, which can be used to identified the original object model with texture. 

For the ProciGen dataset, we used object models from [ShapeNet](https://shapenet.org/), [Objaverse]() and [ABO]() datasets. You can identify the orignal model from these datasets similarly:
- **ABO shapes**: Sequences using this dataset can be identified as the names with suffix containing `abo`. The `ins_name` is the `3dmodel_id` from ABO dataset, which is a unique id to identify the object model. 
- **Objaverse shapes**: The following categories are from objaverse: backpack, basketball, all boxes, stool, suitcase, yoga ball. Similar to ABO, the `ins_name` is the `uid` in Objaverse which can be directly used to download the object model. 
- **Shapenet shapes**: Besides ABO and Objaverse, all other sequences use shapes from ShapeNet. The ShapeNet model can be identified as `<ShapeNet_root>/<synset_id>/<ins_name>/models`. 

**Using the object pose:** given object point p at the canonical shape space, it can be transformed to the current interaction space by simply `p'=Rp + t`, where `R, t` are from entries `rot, trans` stored in the `<object_name>_fit.pkl` file. 

### Camera Parameters
We use the camera parameters from BEHAVE and InterCap to render the synthesized interaction. The camera intrinsic and extrinsic are saved in `<sequence>/info.json` file. 
For more details about reading and loading, please check `data.kinect_transform.ProciGenCameras`. 

## Synthesize ProciGen
Coming soon...

## License
Please see [LICENSE](./LICENSE).

## Citation
If you use the data or code, please cite:
```
@inproceedings{xie2023template_free,
    title = {Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
}
```
Our dataset leverages these datasets, please also consider cite them:
```
@inproceedings{bhatnagar22behave,
    title = {BEHAVE: Dataset and Method for Tracking Human Object Interactions},
    author = {Bhatnagar, Bharat Lal and  Xie, Xianghui and Petrov, Ilya and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022},
}
@inproceedings{huang2022intercap,
    title        = {{InterCap}: {J}oint Markerless {3D} Tracking of Humans and Objects in Interaction},
    author       = {Huang, Yinghao and Taheri, Omid and Black, Michael J. and Tzionas, Dimitrios},
    booktitle    = {{German Conference on Pattern Recognition (GCPR)}},
    volume       = {13485},
    pages        = {281--299},
    year         = {2022}, 
    organization = {Springer},
    series       = {Lecture Notes in Computer Science}
}

@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
@article{collins2022abo,
  title={ABO: Dataset and Benchmarks for Real-World 3D Object Understanding},
  author={Collins, Jasmine and Goel, Shubham and Deng, Kenan and Luthra, Achleshwar and
          Xu, Leon and Gundogdu, Erhan and Zhang, Xi and Yago Vicente, Tomas F and
          Dideriksen, Thomas and Arora, Himanshu and Guillaumin, Matthieu and
          Malik, Jitendra},
  journal={CVPR},
  year={2022}
}
@article{objaverse,
  title={Objaverse: A Universe of Annotated 3D Objects},
  author={Matt Deitke and Dustin Schwenk and Jordi Salvador and Luca Weihs and
          Oscar Michel and Eli VanderBilt and Ludwig Schmidt and
          Kiana Ehsani and Aniruddha Kembhavi and Ali Farhadi},
  journal={arXiv preprint arXiv:2212.08051},
  year={2022}
}
@inproceedings{bhatnagar2019mgn,
    title = {Multi-Garment Net: Learning to Dress 3D People from Images},
    author = {Bhatnagar, Bharat Lal and Tiwari, Garvita and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
    month = {oct},
    organization = {{IEEE}},
    year = {2019},
}
```

## Acknowledgements
This project leverages the following excellent works, we thank the authors for open-sourcing their code and data: 

* The [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/) dataset. 
* The [InterCap](https://intercap.is.tue.mpg.de/) dataset.
* The [ShapeNet](https://shapenet.org/) dataset. 
* The [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset. 
* The [Objaverse](https://objaverse.allenai.org/) dataset.
* The [MGN](https://virtualhumans.mpi-inf.mpg.de/mgn/) human scan dataset. 

