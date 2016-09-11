# Generative Model
This paper list is a bit different from others. I'll put some opinion and summary on it. However, to understand the whole paper, you still have to read it by yourself!    
Surely, any pull request or discussion are welcomed!
- ***Improved Techniques for Training GANs*** [[NIPS 2016]](https://arxiv.org/abs/1606.03498)
  - Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
  - [Code](https://github.com/openai/improved-gan) for the paper
  - Feature matching: instead of maximizing the output of discriminator, it's trained to match the feature on an imtermediate layer of discriminator
  - 
- ***Semantic Image Inpainting with Perceptual and Contextual Losses*** [[arXiv 2016]](https://arxiv.org/abs/1607.07539)
  - Raymond Yeh, Chen Chen, Teck Yian Lim, Mark Hasegawa-Johnson, Minh N. Do
  - Semantic inpainting can be viewed as contrained image generation
- ***Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*** [[ICLR 2016]](https://arxiv.org/abs/1511.06434)
  - Alec Radford, Luke Metz, Soumith Chintala
  - Explore the extension of models for deeper generative model
    - all-convolutional layers: to learn upsampling itself
    - eleminate the fully connected layer: increase the model stability but hurt convergence speed
    - use batchnorm: get deep generator to begin learning, preventing from collapsing all sample to single point
    - ReLU activation: for generator, it helps to converge faster and cover the color space. for discriminator, use leaky ReLU
  - Fractionally-strided convolution instead of deconvolution. To see how fractionally-strided conv is, here's the [link](https://github.com/vdumoulin/conv_arithmetic)
  - Want the model to generalize instead of memorize
  - Use the discriminator as feature extractor (laerned unsupervised) and apply it to supervised laerning task. This produces comparable results
  - Official source code: [Torch version](https://github.com/soumith/dcgan.torch), [Theano version](https://github.com/Newmu/dcgan_code)
- ***Generative Adversarial Networks*** [[NIPS 2014]](https://arxiv.org/abs/1406.2661)
  - Scenario: The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency.
  - In other words, D and G play the following two-player minimax game with value function
  - Find Nash equilibrium by gradient descent of D and G
  - Nice post from Eric Jang, [Generative Adversarial Nets in TensorFlow](http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html)
  - Another post about GAN: [Generating Faces with Torch](http://torch.ch/blog/2015/11/13/gan.html)
  - Official source code: [Theano version](https://github.com/goodfeli/adversarial)
- ***Deep multi-scale video prediction beyond mean square error*** [[ICLR 2016]](https://arxiv.org/abs/1511.05440)
  - Original work only use MSECritetrion to minimize the L2(L1) distance, which induce the blurring output. This work propose the GDL (gradient difference loss), which aims to keep the sharp aprt of the image.
  - Adversial training: create two networks(Discriminative ,Generative model). The goals of D is to discriminate whether the image is fake or not. The goals of G is to generate the image not to discriminated by D. => Adversial
  - D model outputs a scalar, while G model outputs an image
  - Use **Multi-scale architecture** to solve the limitation of convolution (kernel size is limited, eg. 3*3)
  - Still immature. Used in UCF101 dataset, due to the fixed background

## Recommended Post  
- [What are some recent and potentially upcoming breakthroughs in deep learning?](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning/answer/Yann-LeCun?srid=nZuy), written by Yann LeCun
- [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/), from Brandon Amos
- [The application of generative model](https://openai.com/blog/generative-models/#going-forward), from OpenAI
