# InPainting4J
 Dl4J Implementation of Impainting

* Project is based on [DL4J-gans](https://github.com/wmeddie/dl4j-gans)
* and UI from  [better-coding-dl4j-tutorial](https://gitlab.com/better-coding.com/public/dl4j-tutorial.git)

 U-Net model as generator

 70x70 PatchGan as discriminator (C64-C128-C256-C512)

![Input Image](https://i.ibb.co/TvstBg1/input5.png)
![Output Image](https://i.ibb.co/qkGnQJs/image0.gif)

each frame of the gif is equal to 100 iterations it took ~3700 to get clean output.

![Training](https://i.ibb.co/R3W099J/Annotation-2020-05-16-151520.jpg)

If You are planing using Nvidia graphic card, make sure to edit build.gradle
by inserting yours cuda_version and setting dl4j_use_cuda to "true"
