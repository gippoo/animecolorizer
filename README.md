# animecolorizer

## Goal:
Train a model that can colorize a grayscale anime face image.

## Code:
https://github.com/gippoo/animecolorizer/blob/master/colorizer.py

## Method:
**1. Obtain Dataset**

https://www.kaggle.com/mylesoneill/tagged-anime-illustrations<br/>
Used the anime face character dataset. Images are 96x96.

**2. Cleaning**

Manually cleaned data by deleteing images that were false positives (not actually having blue hair or images where no face was present).<br/>
Ended up with 462 images. 412 used for training. 50 held out for validation.<br/>

Sample training images:

![TrainingImgs](https://github.com/gippoo/animecolorizer/blob/master/trainingimgs.png)

**3. Creating Training Data**

Convert images to grayscale. This serves as the input to the network. (Input size: 96x96x1)<br/>
The original color image is left as the target output. (Output size: 96x96x3)<br/>


**4. Model Architecture**

U-net from https://github.com/4g/unet-color/blob/master/colorizer_nongan.py<br/>
Slight differences:<br/>
-Used ReLU instead of leakyReLU<br/>
-No batch norm<br/>
-Sigmoid activation function in last layer (b/c color values were normalized to be between 0 and 1 instead of -1 to 1)

**5. Training the Model**

Loss function: MAE<br/>
Optimizer: RMSProp<br/>
Epochs: 60<br/>
Batch Size: 8<br/>

Final training loss: 0.0477<br/>
Final validation loss: 0.0591

Loss did not seem to have fully converged upon training completion.

**6. Examine Validation Results**

Left: Grayscale image sent as input to the model<br/>
Middle: Output from model<br/>
Right: Original image<br/>

![ValImgs](https://github.com/gippoo/animecolorizer/blob/master/valimgs.png)

The model does a decent job of coloring the grayscale images.

## Apply Model to New Images

We can now take the model to new images outside of the training and validation data.<br/>
Left: Grayscale image sent as input to the model<br/>
Middle: Output from model<br/>
Right: Original image<br/>

![TestImgs](https://github.com/gippoo/animecolorizer/blob/master/testimgs.png)

Note that the model never gets to see the original images in these test cases; it only sees the grayscale versions.
