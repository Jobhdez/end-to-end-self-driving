# end-to-end-self-driving

a neural network that maps images to steering angles, inspired by Nvidia's [paper](https://arxiv.org/pdf/1604.07316v1.pdf) and open source software.

## Model
I have trained a transfer model on 5,000 images and their corresponding steering angles; it may work on bigger datasets.


## Thanks
Here is the [original-project](https://github.com/SullyChen/Autopilot-TensorFlow).
I was inspired by this [project](https://github.com/mankadronit/SelfDrivingCar).
You can get the dataset [here](https://github.com/SullyChen/driving-datasets).

## Performance 
RMSE values for Model 1(ie SGD optimizer): ```training data: 1.66; testing_data: 2.49```

RMSE values for Model 2: ```training: .042; testing: .732```
- used ADAM as an optimizer, added another Dense layer, and the model was trained with 5000 images and their respective steering angles

Based on this [article](http://cs229.stanford.edu/proj2016/report/BoutonHeyse-End-to-endDrivingControlsPredictionsFromImages.pdf) I may be able to
improve the performance of the model by using ADAM as an optimizer and PReLU as an activation function. 
  
