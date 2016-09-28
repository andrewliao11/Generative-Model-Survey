# Generative Model
This paper list is a bit different from others. I'll put some opinion and summary on it. However, to understand the whole paper, you still have to read it by yourself!    
Surely, any pull request or discussion are welcomed!

## Paper
- ***Neural Photo Editing with Introspective Adversarial Networks*** [[arXiv 2016]](http://arxiv.org/abs/1609.07093)
  - Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston
  - Produce specific semantic changes in the output image by use of a **contextual paintbrush** :art: that indirectly modifies the latent vector
  - Hybridization of the Generative Adversarial Network and the  Variational Autoencoder designed for use in the editor, aka IAN
    - Why combine these two? Training VAE is more stable (I guess)  
  - Combine the encoder part of auto-encoder with the discriminator :point_right: discriminator learns a hierarchy of features that are useful for multiple tasks, including inferring latents(encoder in auto-encoder) and comparing samples(D in GAN) 
  - Introspective Adversarial Networks: 
    - generator: generate image that fool the desciminator
    - auto-encoder: to reconstruct the image (image -> feature -> image)
    - dixcriminator: indtead of binary labels, the model is assigned to discriminate the orginal image, reconstructed image, generated image. 
  - IANs maintain the **balance of power** between the generator and the discriminator. In particular, we found that if we made the discriminator too expressive it would quickly out-learn the generator and achieve near-perfect accuracy, resulting in a significant slow-down in training. We thus maintain an “improvement ratio” rule of thumb, where every layer we add to the discriminator was
accompanied by an addition of three layers in the generator.
- ***WaveNet: A Generative Model for Raw Audio*** [[arXiv 2016]](http://128.84.21.199/abs/1609.03499)
  - Aaron van den oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
  - Not a GAN architecture
  - [DeepMind Post](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
  - Use generative model to automatically generate the audio
  - Fully dilated causal convolution
    - causal: only depends on the previous sample data
    - dilated: use dilated convolution to increase the receptive field
  - **Conditional wavenet**
    - Global condition: use a V matrix as projection matrix (See eq.2)
    - Local condition: this time the additional input can be a sequence. Use a transpose convolutional network to project(upsample) it to the same length as input audio signal.
- ***Improved Techniques for Training GANs*** [[NIPS 2016]](https://arxiv.org/abs/1606.03498)
  - Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
  - [Code](https://github.com/openai/improved-gan) for the paper
  - Feature matching: instead of maximizing the output of discriminator, it's trained to match the feature on an imtermediate layer of discriminator
  - Minibatch-discrimination: 
    - Motivation: because the discriminator processes each example **independently**, there is no coordination between its gradients, and thus no mechanism to tell the outputs of the generator to become more dissimilar to each other
    - Allow the discriminator to look at **multiple data examples in combination-**, and perform what we call minibatch discrimination
    - Calculate the l1-error btn each samples feature and finally concatenate the output with the sample feature
    - **Hope the generated images to be diverse** :point_right: less probability to collapse
  - Historical averaging to stablize the training process
  - **Automatic evaluation metrix**, which is based on the inception model (See section 4)
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

## Suggest papers
- ***Adversarial examples in the physical world*** [[arXiv 2016]](https://arxiv.org/abs/1607.02533)
  - Alexey Kurakin, Ian Goodfellow, Samy Bengio  
- ***Generative Visual Manipulation on the Natural Image Manifold*** [[ECCV 2016]](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/eccv16_gvm.pdf)
  - Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros
  - [Demo video](https://www.youtube.com/watch?v=9c4z6YsBGQ0)
- ***Attend, Infer, Repeat: Fast Scene Understanding with Generative Models*** [[NIPS 2016]](http://arxiv.org/abs/1603.08575)
  - S. M. Ali Eslami, Nicolas Heess, Theophane Weber, Yuval Tassa, David Szepesvari, Koray Kavukcuoglu, Geoffrey E. Hinton

## Recommended Post  
- [What are some recent and potentially upcoming breakthroughs in deep learning?](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning/answer/Yann-LeCun?srid=nZuy), written by Yann LeCun
- [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/), from Brandon Amos
- [The application of generative model](https://openai.com/blog/generative-models/#going-forward), from OpenAI
- [Autoencoders](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/), it's important to know the difference between auto-encoder and generatice model
