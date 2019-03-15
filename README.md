
# Reinforcement Learning on DonkeyCar Simulator

## Dependencies
- donkey-gym 0.1
- gym 0.10.9
- keras 1.2.0
- numpy 1.16.2
- opencv3 3.1.0
- scikit-image 0.12.3
- tensorboard 1.13.1
- tensorflow 1.13.1

Although there was a small issue when using tensorflow 1.12.0, it still runs fine. 
The only issue is that you cannot download the SVG files from tensorboard. 

To upgrade your tensorflow, follow the instructions below.(If you upgrade tensorflow, you also need to upgrade numpy to 1.16.2 as well.)

```pip install tensorflow --upgrade```


## Monitor Progress
Here I use tensorboard to monitor the learning progress as bellow.

[image1]: https://github.com/zenyanagata/ReinforceDonkeyCar/blob/master/util/tensorboard_example.PNG
![alt text][image1]

We monitor **Epsilon**, **Loss**, and **Reward** derived each episode. This enables us to make the most of every training transaction.

To use tensorboard follow the instructions below. Make sure you've passed the abs pass to ```--ligdir```.

```tensorboard --logdir=C:\Users\~~~\dqn\logs\deep_q_network```

## Legal Notice
donkey-gym is under MIT license, authored by Tawn Kramer and original sources located [here](https://github.com/tawnkramer/donkey_gym/).

Other packages authered by Zenya Nagata. For citation email (zenya.nagata@outlook.jp).
