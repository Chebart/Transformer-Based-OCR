# Transformer-OCR
<p align="center">
  <img src="https://github.com/Chebart/Transformer-OCR/assets/88379173/55d6c388-bab2-4bb2-b640-8ad7e88d198a">
</p>


Implementation of transformer for optical character recognition of russian words.

Text recognition is a long-standing research problem for document digitalization. Existing approaches are usually built
based on CNN for image understanding and RNN for charlevel text generation. In addition, another language model is
usually needed to improve the overall accuracy as a postprocessing step. This implementation leverages
the Transformer architecture for both image understanding and wordpiece-level text generation.

## Useful links

1. Li M. et al. [Trocr: Transformer-based optical character recognition with pre-trained models](https://arxiv.org/pdf/2109.10282.pdf)
		 //arXiv preprint arXiv:2109.10282. – 2021.

2. Atienza R. [Vision transformer for fast and efficient scene text recognition](https://arxiv.org/pdf/2105.08582.pdf) 
    //Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part I 16. – Springer International Publishing, 2021. – С. 319-334.
	
3. Kim G. et al. [Ocr-free document understanding transformer](https://arxiv.org/pdf/2105.08582.pdf) 
     //Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII. – Cham : Springer Nature Switzerland, 2022. – С. 498-517.


## Python packages
```
pytorch==1.13.1+cu117
torchvision==0.14.1+cu117
datasets==2.10.1
transformers==4.27.3
```
