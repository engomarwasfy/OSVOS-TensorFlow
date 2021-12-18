from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)

# User defined parameters
'''
seq_names = ["bear","dance-jump","horsejump-low",
"rhino","blackswan","dance-twirl", "kite-surf"	,"rollerblade","bmx-bumps","dog",
"kite-walk","scooter-black","bmx-trees","dog-agility","libby","scooter-gray",
"boat","drift-chicane","lucia","soapbox","breakdance","drift-straight","mallard-fly","soccerball",
"breakdance-flare","drift-turn","mallard-water","stroller",
"bus","elephant","motocross-bumps","surf",
"camel","flamingo","motocross-jump","swing",
"car-roundabout","goat","motorbike","tennis",
"car-shadow","hike","paragliding","train",
"car-turn","hockey","paragliding-launch",
"cows","horsejump-high","parkour"]
'''

seq_names = ["aerobatics","dog-agility","lab-coat","rollercoaster",
"bear","dog-gooses","lady-running","salsa",
"bike-packing","dogs-jump","libby","schoolgirls",
"blackswan","dogs-scale","lindy-hop","scooter-black",
"bmx-bumps","drift-chicane","loading","scooter-board",
"bmx-trees","drift-straight","lock","scooter-gray",
"boat","drift-turn","longboard","seasnake",
"boxing-fisheye","drone","lucia","sheep",
"breakdance","elephant","mallard-fly","shooting",
"breakdance-flare","flamingo","mallard-water","skate-jump",
"bus","giant-slalom","man-bike","skate-park",
"camel","girl-dog","mbike-trick","slackline",
"carousel","goat","miami-surf","snowboard",
"car-race","gold-fish","monkeys-trees","soapbox",
"car-roundabout","golf","motocross-bumps","soccerball",
"car-shadow","guitar-violin","motocross-jump","stroller",
"car-turn","gym","motorbike","stunt",
"cat-girl","helicopter","mtb-race","subway",
"cats-car","hike","night-race","surf",
"chamaleon"	 , "hockey"	,   "orchid"	  ,"swing",
"classic-car"	,"horsejump-high" ,"paragliding","tandem",
"color-run",	  "horsejump-low", "paragliding-launch",  "tennis",
"cows","horsejump-stick","parkour","tennis-vest",
"crossing","hoverboard"	,"people-sunset","tractor",
"dance-jump"	,"india" ,"pigs"		,       "tractor-sand",
"dance-twirl","judo","planes-crossing" ,"train",
"dancing","kid-football","planes-water","tuk-tuk",
"deer","kite-surf","rallye","upside-down",
"disc-jockey","kite-walk"	,"rhino","varanus-cage",
"dog","koala","rollerblade","walking"]



length = len(seq_names)
gpu_id = 0
train_model = True
max_training_iters = 10

# Show results
overlay_color = [255, 0, 0]
transparency = 0.6
for i, seq_name in enumerate(seq_names, start=1):
  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"+str(i)+"."+seq_name+" from "+str(length))
  result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)

  # Train parameters

  parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
  logs_path = os.path.join('models', seq_name)
  # Define Dataset
  test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
  test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
  if train_model:
      train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                    os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
      dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
  else:
      dataset = Dataset(None, test_imgs, './')

  # Train the network
  if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
      with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        osvos.train_finetune(
            dataset,
            parent_path,
            side_supervision,
            learning_rate,
            logs_path,
            save_step,
            save_step,
            display_step,
            global_step,
            iter_mean_grad=1,
            ckpt_name=seq_name,
        )
  # Test the network
  with tf.Graph().as_default():
      with tf.device('/gpu:' + str(gpu_id)):
          checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
          osvos.test(dataset, checkpoint_path, result_path)
  plt.ion()
  for img_p in test_frames:
      frame_num = img_p.split('.')[0]
      img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
      mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
      mask = mask//np.max(mask)
      im_over = np.ndarray(img.shape)
      im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
      im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
      im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
      plt.imshow(im_over.astype(np.uint8))
      plt.axis('off')
      plt.show()
      plt.pause(0.01)
      plt.clf()
