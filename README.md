# MSF-CloudNet<br>
## A network based on deep learning for cloud detection in remote sensing, the paper is 《 Cloud Detection in Remote Sensing Images Based on Multiscale Features-Convolutional Neural Network[J]. 》<br>
* loader_lc8.py <br>
provides classe for loading landsat8-biome datasets.<br>
* saver_loader.py <br>
provides the class and method of saving model and loading pre training model.<br>
* metrics.py <br>
provides some evaluation indexes for model validation.<br>
* loss.py <br>
provides the loss function used in the back propagation of training. There are two kinds of loss functions, namely "CE" and "focal".<br>
* Network.py <br>
defines the architecture of the neural network, which is named MSNetwork.<br>
* train.py and valid.py <br>
are the code of network training and validation respectively. The dataset used is 12 images in landsat8-biome, which is divided into 256 size patches, of which the training dataset accounts for 0.8, and the test dataset accounts for 0.2, and the data is scrambled during training.<br><br><br><br>



### Quote:Shao Z , Pan Y , Diao C , et al. Cloud Detection in Remote Sensing Images Based on Multiscale Features-Convolutional Neural Network[J]. IEEE Transactions on Geoence and Remote Sensing, 2019, PP(99):1-15.
