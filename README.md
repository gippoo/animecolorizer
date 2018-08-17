# animecolorizer

## Method:
**1. Obtain Dataset**

https://www.kaggle.com/mylesoneill/tagged-anime-illustrations<br/>
Images are 96x96

**2. Cleaning**

Manually cleaned data by deleteing images that were false positives (not actually having blue hair)<br/>
Ended up with 462 images. 412 used for training. 50 held out for validation.<br/>

Sample training images:

![TrainingImgs](https://github.com/gippoo/animecolorizer/blob/master/trainingimgs.png)

**3. Creating Training Data**

Convert images to grayscale. This serves as the input to the network. (Input size: 96x96x1)<br/>
The original color image is left as the target output. (Output size: 96x96x3)<br/>


**4. Model Architecture**

U-net from https://github.com/4g/unet-color/blob/master/colorizer_nongan.py<br/>
Slight differences:<br/>
-Used relu instead of leakyrelu<br/>
-No batch norm

**5. Training the Model**

Loss function: MAE<br/>
Optimizer: RMSProp<br/>
Epochs: 60<br/>
Batch Size: 8<br/>

**6. Applying the Model to New Images**

Left: Grayscale image sent as input to the model<br/>
Middle: Output from model<br/>
Right: Original image<br/>

![TestImgs](https://github.com/gippoo/animecolorizer/blob/master/testimgs.png)
