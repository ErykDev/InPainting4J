# InPainting4J
 Dl4J Implementation of Inpainting

* Project is based on [DL4J-gans](https://github.com/wmeddie/dl4j-gans)
* UI from  [better-coding-dl4j-tutorial](https://gitlab.com/better-coding.com/public/dl4j-tutorial.git)

 U-Net model as generator

 70x70 PatchGan as discriminator (C64-C128-C256-C512)

![Input Image](/imgs/input.png)
![Output Image](/imgs/output.gif)

each frame of the gif is equal to 50 iterations it took ~3000 to get clean output.

![Training](/imgs/loss.png)

If You are planing using Nvidia graphic card, make sure to edit build.gradle
by inserting yours cuda_version and setting dl4j_use_cuda to "true"

Ready to use models:
https://drive.google.com/drive/folders/1CXltL6oZSbfCGpEhyGv77eA0IqWQSRuD?usp=sharing

to use them just put the models into project folder.
