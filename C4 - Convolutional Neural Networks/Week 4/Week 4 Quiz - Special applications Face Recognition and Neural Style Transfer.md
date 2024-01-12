## Week 4 quiz - Special applications: Face recognition & Neural style transfer

1. Face verification requires comparing a new picture against one person’s face, whereas face recognition requires comparing a new picture against K person’s faces.

	> True

2. Why do we learn a function d(img1,img2) for face verification? (Select all that apply.)

	> This allows us to learn to recognize a new person given just a single image of that person.

	> We need to solve a one-shot learning problem.

3. In order to train the parameters of a face recognition system, it would be reasonable to use a training set comprising 100,000 pictures of 100,000 different persons.

	> False

4. Which of the following is a correct definition of the triplet loss? Consider that α>0. (We encourage you to figure out the answer from first principles, rather than just refer to the lecture.)

	> ```max(||f(A)−f(P)||^2 − ||f(A)−f(N)||^2 + α, 0)```


5. Consider the following Siamese network architecture: The upper and lower neural networks have different input images, but have exactly the same parameters.

	> True

6. You train a ConvNet on a dataset with 100 different classes. You wonder if you can find a hidden unit which responds strongly to pictures of cats. (I.e., a neuron so that, of all the input/training images that strongly activate that neuron, the majority are cat pictures.) You are more likely to find this unit in layer 4 of the network than in layer 1.

	> True

7. Neural style transfer is trained as a supervised learning task in which the goal is to input two images (x), and train a network to output a new, synthesized image (y).

	> False

8. In the deeper layers of a ConvNet, each channel corresponds to a different feature detector. The style matrix G[l] measures the degree to which the activations of different feature detectors in layer l vary (or correlate) together with each other.

	> True

9. In neural style transfer, what is updated in each iteration of the optimization algorithm?

> The pixel values of the generated image G

10. You are working with 3D data. You are building a network layer whose input volume has size 32x32x32x16 (this volume has 16 channels), and applies convolutions with 32 filters of dimension 3x3x3 (no padding, stride 1). What is the resulting output volume?

	> ```30 * 30 * 30 * 32```

11. You are working with 3D data. The input image has 64X64X64X3. if you apply convolution layer with 16 filters of size 4x 4 x 4, zero padding and size 2. What is the out put volume ?
	> 31 x 31 x 31 x 16

12. In neural style transfer, which of the following better express the gradients used?

     	>  dW/ dG
14. Neural style transfer uses images Content C, styles S, The loss function used to generate image G is composed of which of the following "

 	> J(style) that compares S and g.
  	> J(content) that compares C and G
16. Triplet los is larger in which of the following cases?

 	> When the encoding of A is closer to the encoding of N, than to the encoding of P
18. Which of the folowing do you agree with ?
    	> It will be more efficient to learn a function d(img1, img2) for this task.
    	> This can be considered a one-shot learning task
