# animecolorizer

## Method:
**1. Obtain Dataset**
https://www.kaggle.com/mylesoneill/tagged-anime-illustrations
images are 96x96

**2. Cleaning**
Manually cleaned data by deleteing images that were false positives (not actually having blue hair)
Ended up with 462 images. 412 used for training. 50 held out for validation.

**3. Creating Training Data**
Convert images with the 'blue_hair' tag to grayscale. This serves as the input to the network. (Input size: 96x96x1)
The original color image is left as the target output. (Output size: 96x96x3)

Sample training images:

![TrainingImgs](https://github.com/gippoo/animecolorizer/blob/master/trainingimgs.png)

**4. Model Architecture**
U-net from https://github.com/4g/unet-color/blob/master/colorizer_nongan.py
Slight differences: 
-Used relu instead of leakyrelu 
-No batch norm

**5. Training the Model**
Loss function: MAE
Optimizer: RMSProp
Epochs: 60
Batch Size: 8

**6. Applying the Model to New Images**
Left: Grayscale image sent as input to the model
Middle: Output from model
Right: Original image

![TestImgs](https://github.com/gippoo/animecolorizer/blob/master/testimgs.png)
