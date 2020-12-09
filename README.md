# end-to-end-self-driving

a neural network that maps images to steering angles, inspired by Nvidia's [paper](https://arxiv.org/pdf/1604.07316v1.pdf) and open source software.

## Model
I have trained a transfer model on 5,000 images and their corresponding steering angles; it may work on bigger datasets.


## Thanks
Here is the [original-project](https://github.com/SullyChen/Autopilot-TensorFlow).
I was inspired by this [project](https://github.com/mankadronit/SelfDrivingCar).
You can get the dataset [here](https://github.com/SullyChen/driving-datasets).

## Performance 
RMSE values for Model 1(ie SGD optimizer): ```training: 1.66; validation: 2.49```

RMSE values for Model 2: ```training: .042; testing: .732```
- used ADAM as an optimizer, added another Dense layer, and the model was trained with 5000 images and their respective steering angles

RMSE vlaues for Model 3: ```training .072; validation: .22```
- all I did differently was use a generator that allows me work with 40,000+ images and their respective steering angles; nevertheless, I only used 10k images 
and their respective steering angles. By adding more data the, obviously, the model generalizes better. So, if you play around with this program and want to improve the model just work with whole dataset and, if needed, regularize it by adding dropout layers to the dense layers except for the output layer. According to this [paper](https://arxiv.org/pdf/1803.08450.pdf) this type of architecture may work. 

  
