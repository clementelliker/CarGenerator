## CarGenerator

This project aims at creating a car image generator which can be conditionned by a continuous feature like the sportivity. This was made using [CcGan Paper](https://openreview.net/pdf?id=PrzjugOsDeE) and [CcGan Repository](https://github.com/UBCDingXin/improved_CcGAN).

<br/><br/>

# How to use:

-Download this [Dataset](https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset)

-Extract it in Data/Cars

-If you want to use only the images with no background (which I advise you to), use preprocessing.ipynb to generate the correct dataset

-Optionnal: if you want to generate images with other labels than color/RPM use the embeding(_nobg).ipynb to get the label projection model y2h

-Run ccGan(dim).ipynb to start your training with the option you chose

<br/><br/>

# Notes:

-the color labels are only for the no_background dataset
-the y2h network given are only for 128*128 images

<br/><br/>

# Obtained results:

![Example of cars generated](https://github.com/clementelliker/CarGenerator/blob/main/images/ex64.PNG?raw=true "Title")
<br/>
A collumn corresponds to the same generating seed with different conditionnal values (0 to 1 top to bottom). On the left conditionned by RPM and on the right conditionned by luminosity.
