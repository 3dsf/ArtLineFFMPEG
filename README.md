[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3dsf/ArtLineFFMPEG/blob/main/ArtLineFFMPEG.ipynb)

# ArtLineFFMPEG
An Implementation of https://github.com/vijishmadhavan/ArtLine

--- 

Allows your to trasform youtube videos to ArtLine using Google Colab or locally.  

---  

### **Conda Enviroment** for local execution of either the **Jupyter Notebook** or **run.py** : 
- `conda create -n artLine-GPU python=3.7 pytorch=1.7 torchvision cudatoolkit=10.1 fastai=1.0 opencv ffmpeg==4.0.2 ffmpeg-python wget x264 -c pytorch -c fastai -c conda-forge`

---  
---  

## **run.py** 
- processes video locally 
- uses a **buffer** and is **quicker** because of that
- it can also process folders of images
