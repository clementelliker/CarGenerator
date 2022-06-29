# CarGenerator

This project aims at creating a car image generator which can be conditionned by a continuous feature like the sportivity. This was made using [CcGan Paper](https://openreview.net/pdf?id=PrzjugOsDeE) and [CcGan Repository](https://github.com/UBCDingXin/improved_CcGAN).

How to use:

-Download this [Dataset](https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset)

-Extract it in Data/Cars

-If you want to use only the images with no background (which I advise you to), use preprocessing.ipynb to generate the correct dataset

-Optionnal: if you want to generate images with other labels than color/RPM use the embeding(_nobg).ipynb to get the label projection model y2h

-Run ccGan(dim).ipynb to create your generator with the option you chose


Notes:

-the color labels are only for the no_background dataset
-the y2h network given are only for 128*128 images


Obtained results:

