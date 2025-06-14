## Week 3 quiz - Detection algorithms

1. You are building a 3-class object classification and localization algorithm. The classes are: pedestrian (c=1), car (c=2), motorcycle (c=3). What would be the label for the following image? Recall y=[pc,bx,by,bh,bw,c1,c2,c3]

	> ```y=[1,0.3,0.7,0.3,0.3,0,1,0]```

2. Continuing from the previous problem, what should y be for the image below? Remember that “?” means “don’t care”, which means that the neural network loss function won’t care what the neural network gives for that component of the output. As before, y=[pc,bx,by,bh,bw,c1,c2,c3].

	> ```y=[0,?,?,?,?,?,?,?]```

3. You are working on a factory automation task. Your system will see a can of soft-drink coming down a conveyor belt, and you want it to take a picture and decide whether (i) there is a soft-drink can in the image, and if so (ii) its bounding box. Since the soft-drink can is round, the bounding box is always square, and the soft drink can always appears as the same size in the image. There is at most one soft drink can in each image. Here’re some typical images in your training set: What is the most appropriate set of output units for your neural network?

	>  Logistic unit, bx, by

4. If you build a neural network that inputs a picture of a person’s face and outputs N landmarks on the face (assume the input image always contains exactly one face), how many output units will the network have?

	> 2N

5. When training one of the object detection systems described in lecture, you need a training set that contains many pictures of the object(s) you wish to detect. However, bounding boxes do not need to be provided in the training set, since the algorithm can learn to detect the objects by itself.

	> False

6. Suppose you are applying a sliding windows classifier (non-convolutional implementation). Increasing the stride would tend to increase accuracy, but decrease computational cost.

	> False

7. In the YOLO algorithm, at training time, only one cell ---the one containing the center/midpoint of an object--- is responsible for detecting this object.

	> True

8. What is the IoU between these two boxes? The upper-left box is 2x2, and the lower-right box is 2x3. The overlapping region is 1x1.

	> 1/9

9. Suppose you run non-max suppression on the predicted boxes above. The parameters you use for non-max suppression are that boxes with probability ≤ 0.4 are discarded, and the IoU threshold for deciding if two boxes overlap is 0.5. How many boxes will remain after non-max suppression?

	> 5

10. Suppose you are using YOLO on a 19x19 grid, on a detection problem with 20 classes, and with 5 anchor boxes. During training, for each image you will need to construct an output volume y as the target value for the neural network; this corresponds to the last layer of the neural network. (y may include some “?”, or “don’t cares”). What is the dimension of this output volume?
	
	> 19x19x(5x25)

11. The most adequate output for a network to do the required task is y=[p_c, b_x, b_y,b_h, b_w, c_1](Which of the following do you agree with the most?)
    
	> False, we don't need b_h, b_w  since the cans are all the same size.
 
13. When building a neural network that inputs a picture of a person's face and outputs N landmarks on the face (assume that the input image contains exactly one face), which is true about y_hat(i)?
 
	> y_hat(i) has a shape of (2N,1)
 
14. Semantic segmentation can only be applied to classify pixels of images in a binary way as 1 or 0, according to whether they belong to a certain class or not. True/False?
    
 	> False
  
16. Using the concept of Transpose Convolution, fill in the values of X, Y and Z below. (padding =1, stride =2),input = [1 2],[3 4] Filter= [1 0 -1], [1 0 -1], [1, 0 ,-1]
    
 	> X =2 , Y =-6, Z=-4
  
18. When using the U-Net architecture with an input h X w X c,  where c denotes the number of channels, the output will always have the shape h X W X c ?
	> False

19. When using the U-Net architecture with an input h X w X c,  where c denotes the number of channels, the output will always have the shape h X W  ?
	> False
 	>  The output of the U-Net architecture can be h × w × k, where k is the number of classes. The number of channels doesn't have to match between input and output.

20. Which of the following do you agree about the use of anchor boxes of YOLO ?
    
 	> Each object is assigned to the grid cell that contains the object midpoint
 	> Each object is assigned to an anchor box with the highest IoU inside the assigned cell.

23. When trying to build a system that assigns a value of 1 to each pixel that is part of a tumor from a medical image taken from a patient. This is a problem of localisation ?
    
 	> False
  

26. If we use anchor boxes in YOLO, we no longer need the cordinates of the bounding box. Since they are given by the cell position of the grid and anchor box selection.

	 	> False

28. 





 
