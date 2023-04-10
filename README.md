# Macro_to_Micro
Official Pytorch code for the paper:        
**Macro to Micro: Brain MR Image-to-Image Translation between Structural MRI, Diffusion MRI, and Tractography modalities** (working title)

Authors:       
- 김수영 (Sooyoung Kim; rlatndud0513@snu.ac.kr) Graduate school student at Seoul National University (co-first author)
- 권준우 (Joonwoo Kwon; pioneers@snu.ac.kr) Graduate school student at Seoul National University (co-first author) <br>
- 차지욱 (Jiook Cha; cha.jiook@gmail.com) Assistant Professor at Seoul National University (corresponding author) <br>

## Overview
<p align="center"><img width="1065" alt="overall" src="https://user-images.githubusercontent.com/43199011/230920595-4b3a85a2-d8ec-443b-9ad9-eac556a8163c.png"></p>   
We are generating **MR Image Translation** framework, which can save time and cost for getting multi-modality MR Images. Also, we can prove that macroscale structure brain information can generate microscale structure brain features. <br>

Input of the framework is 3D T1-weighted images, structural MRI, the intermediate output is Diffusion Tensor Image (DTI), diffusion MRI, and output is 3D Tractography. T1 is the most used modality to get brain structure. However, diffusion MRI is not well used and it needs adaptive time and cost. Also, when we have diffusion-weighted image using MR, we have to preprocessing to get DTI and tractography which process takes almost 12 hours for each subject. <br>

As our Image-to-Image Translation model use only 2D image data, we use the slices of 3D T1 images as model input. In phase1, the model translate the T1 images to DTI. In phase2, we generate 2D tractography slice images from DTI using I2I. In phase3, we render 3D tractography using 2D tractogram slice images using NeRF.

<p align="center"></p>  
In phase1, we use Image-to-Image Translation model using encoder, generator, discriminator, and patch-discriminator. 
   
Phase1 model is on hyper-parameter tunning and the model for phase2 is currently being designed. Phase3 NeRF is also on desining.

## Current Result
<p align="center"></p>   
