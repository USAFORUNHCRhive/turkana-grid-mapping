# Sample Tutorial
This project supports mapping electrical distribution infrastructure from high resolution drone imagery. Electrical distribution infrastructure
mapping is performed in two parts:
1. [Detecting Electrical Poles](#pole-detection)
2. [Segmenting Electical Lines](#line-segmentation)

Practical [tips](#tips) are also provided to ensure proper replication of the work. Access and download sample data provided by USA for UNHCR and the Humanitarian OpenStreetMap Team, as described in the [README.md](README.md).


## 1. Pole Detection
Pole detection is performed using a [Point supervision](https://arxiv.org/abs/1807.09856) method for localizing and counting points. 
Here we train a Fully Convolutional Network(FCN-8) to predict poles as blobs, where input poles point locations are used to provide a supervisory
learning signal. Below are the steps to obtain poles as points once you are in the *src/poles/* directory. 

### 1.1. Data prep
- Download and place your train and val images in a base directory. Adjust the ```pole_config.yml``` file
*root_dir* to reflect the name of your base directory.
- The model can be trained by generating the label rasters on the fly or by utilizing a pre-generated raster label dataset.
Use the line of code below to generate a raster label dataset given a set of pole labels as points stored in a vector file. Note that the pole vector
labels path should be included in ```pole_config.yml```, *pole_vector_file* variable.

```
python data/pole_dataprep.py
```

### 1.2. Model training & inference
All parameters for model training and inference can be modified in the ```pole_config.yml```. 
For inference specify the test image directory and where to store predicted masks in the yaml file.
Use the following lines to train the model and run inference. 

```
python pole_train.py --method_name <EXPERIMENT_NAME> --gpu <GPU>
python pole_inference.py --method_name <EXPERIMENT_NAME> --gpu <GPU>
```

### 1.3. Post processing & metrics computation
The prediction masks can be post-processed to compute metrics using the line below

```
python poledetect/pole_metrics.py --method_name <EXPERIMENT_NAME> --buffer 10 --noise_threshold 1 --label_fn <LABEL_FN> --image_fn <IMAGE_FN> --preds_fn <PREDICTIONS_FN>
```


## 2. Line Segmentation
Line segmentation is performed at the patch-level instead of the pixel-level. Electrical distribution lines
are very small and thus performing pixel-level segmentation does not produce the best results, especially in the presence of noisy labels.
We perform line segmentation by using an assymetric UNET to support patch-level predictions.
Below are the steps to obtain line segments once you are in the *src/lines/* directory. 

### 2.1. Data preparation
- The line segmentation demo utilizes the same images as the pole detection demo.
In the ```line_config.yml``` file change the *root_dir* to the base directory with corresponding train and val images.
- Generating line segments on-the-fly can be expensive, thus the line segments label masks are generated ahead of time.
The data preparation step is important to convert vector line labels to raster labels. Use the line below to 
generate raster line labels.

```
python data/line_dataprep.py
```

### 2.2. Model training & inference
All parameters for model training and inference can be modified in the ```line_config.yml```. 
The *segm_filter_size*  param determines the patch size for predictions relative to the input image size. For example,
if a *(batch_size x in_channels x 512 x 512)* is input into the model and *segm_filter_size*=8, 
then a (batch_size x n_classes x 64 x 64) output will be obtained from the model. An alternative way to 
think of this is, smaller values for *segm_filter_size* will result in finer lines. 

The model is trained with a weighted cross-entropy loss. Indicate class weights in the config file.

```
python line_train.py --method_name <EXPERIMENT_NAME> --gpu <GPU>
python line_inference.py --method_name <EXPERIMENT_NAME> --gpu <GPU>
```

## 3. Tips

### 3.1. General
- You can use tensorboard to monitor training and visualize predictions using: 
```tensorboard --logdir logs/<EXPERIMENT_NAME>/ --port PORT```.
- If you observe a high number of false positives, include them as hard negatives and retrain the model.
- When creating labels, include a 'type' attribute to indicate *pole* (pole location), *not_pole* (pole hard negatives),
*line* (electrical line segment) and *not_line* (line hard negative).

### 3.2. Pole detection
- Training with a batch size of 1 at a very small learning for long periods gives good results.
- Avoid small patch sizes as the model will predict the full image as a blob. Default is 512.
- We utilize a valid data mask here for the *nodata_check* function. This means we can provide hard.
negative examples (locations that look like poles but are not poles) through the mask. The 
mask is also generated at the data preparatory step or on the fly.
- To generate the labels on-the-fly pass the *pole_vector_file* as an argument to the *datamodule* in 
```pole_train.py```.
- Inspect prediction rasters to find a good noise threshold for filtering points (currently set to 1). 
Apply a similar method to determining an appropriate buffer size (default is 10). Note that the buffer radius determines
a radius to buffer groundtruth points thereby creating a catchment area. If predicted poles fall within the catchment area,
then it is considered a true positive.

### 3.3. Line segmentation
- Class weights are influenced by the output patch size relative to the input patch. The default weights work well
for a *segm_filter_size=8*
- Use the LineUnet model and note that log2(segm_filter_size) should be less than the network depth.