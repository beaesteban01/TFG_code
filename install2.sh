#!/usr/bin/bash
conda create --name tf python=3.6;
conda activate tf;
conda install scipy;
pip install --upgrade sklearn;
pip install --upgrade pandas;
pip install --upgrade pandas-datareader;
pip install --upgrade matplotlib;
pip install --upgrade pillow;
pip install --upgrade tqdm;
pip install --upgrade requests;
pip install --upgrade h5py;
pip install --upgrade pyyaml;
pip install --upgrade psutil;
pip install --upgrade tensorflow==1.12.0;
pip install --upgrade keras==2.2.4;

#Descargar el zip de Julio
wget https://nesg.ugr.es/nesg-ugr16/download/attack/july/week5/july_week5_csv.tar.gz;
mkdir tfg;

tar -xzvf july_week5_csv.tar.gz;
head -6000000 uniq/july.week5.csv.uniqblacklistremoved >> tfg/july_600mb.csv;