
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
pip install tensorflow --upgrade
[//]: # (Image References)

[image1]: ./util/tensorboard_example.png "Tensorboard example"

## Monitor Progress
Here I use tensorboard to monitor the learning progress as bellow.

![alt text][image1]

## Legal Notice

Package is under MIT license. Authored by Tawn Kramer and original sources located [here](https://github.com/tawnkramer/donkey_gym/).
