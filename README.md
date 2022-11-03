# AdverseBiNet-pytorch
 AdverseBiNet-pytorch

This is a reimplementation of AdverseBiNet with pytorch. 
[github] (https://github.com/ankanbhunia/AdverseBiNet)
[paper] (https://arxiv.org/abs/1810.11120v1)


# Improving Document Binarization via Adversarial Noise-Texture Augmentation (ICIP 2019)
![image](https://user-images.githubusercontent.com/102145595/199649484-7f67c7e8-e09b-4d9f-905a-1b9fe8224357.png)

In this paper, we propose a two-stage network that first learns to augment the document images by using neural style transfer technique. For this purpose, we construct a Texture Augmentation Network that transfers the texture element of a degraded reference document image to a clean binary image.
![image](https://user-images.githubusercontent.com/102145595/199648230-9e47cc9f-3b1b-4e41-83af-c0141f598e12.png)
In this way, the network creates multiple versions of the same textual content with various noisy textures, enlarging the available document binarization datasets. At last, the newly generated images are passed through a Binarization network to get back the clean version.











### result in 1 epoch 
![image](https://user-images.githubusercontent.com/102145595/199649465-b5782658-f0cf-4a90-a5f6-17cc5a315710.png)

### will be updated...
