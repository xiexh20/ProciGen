# ProciGen: Procedural Interaction Synthesis 
This documentation provides more details on procedurally synthesize interaction with new shapes. 

For the following, please make sure you have installed the dependencies properly and downloaded the quick start demo data.  

### Render more objects

We use shapes form ShapeNet, Objaverse, and ABO dataset. Depending on your interests, you may download one or many of these datasets:
- ShapeNet: Download from [here](https://shapenet.org/login/). Unzip and change `SHAPENET_ROOT` in `paths.py` to your local path. 
- Objavers: No need to pre-download, just do `pip install objaverse` and objects will be downloaded automatically when running the rendering script. Warning: objaverse downloads are saved to home directory (`~/.objaverse`), 
modify the BASE_PATH in `objaverse.__init__.py` if that is not desired. 
- ABO: Download from [ABO website](https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download). We use only the [3D models](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar). Download, unzip, and modify `ABO_ROOT` in `paths.py` to local path.  

Once the corresponding files are downloaded, you can re-render the corresponding ProciGen sequences using `python render/blender_rerender.py`. Example command to render 5 frames of the monitor sequence (shapes from ShapeNet):
```shell
python render/blender_rerender.py -s /path/to/procigen/Date04_Subxx_monitor_synzv2-01  -fe 5
```

### Synthesize more objects 
**Data**: download the quick start data. We prepare data for more objects, download them from xxx. Once download, replace the demo assets with full assets:
```shell
rm -r example/assets
unzip ProciGen-assets.zip -d example/assets
rm ProciGen-assets.zip
```
With this full assets, you can do synthesize for other objects. Example commands for synthesize and rendering:
```shell
# Use chair from shapenet and interaction from behave chairwood
python synz/synz_batch.py -src shapenet --object_category chair -obj chairblack -s "*chairblack*" -o <your params output>
python render/render_hoi.py -p <your params output> -src objaverse --obj_name chairblack -o <your render output>

# Use table from ABO and interaction from behave tablesmall sequences
python synz/synz_batch.py -src abo --object_category abo-table -obj tablesmall -s "*tablesmall*" -o <your params output>
python render/render_hoi.py -p <your params output> -src abo --obj_name tablesmall -o <your render output>

# Use box from objaverse and interaction from behave boxlarge sequences
python synz/synz_batch.py -src objaverse --object_category box -obj boxlarge -s "*boxlarge*" -o <your params output>
python render/render_hoi.py -p <your params output> -src objaverse --obj_name boxlarge -o <your render output>
```

### Synthesize beyond the shapes used by ProciGen
train AE for your own object shapes, process the shapes. coming soon...

