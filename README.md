
# Understanding Image Advertisements for Predicting Sentiments
Promotion of products is now a common practice and is heavily controlled by broadcasting of advertisements. Image based advertisements are still one of the best ways to promote products but it is painstakingly difficult to personalize the content for the target audience and covey the sentiments. It is proven that an image can be perceived in different manners and hence different emotions can be conveyed via them. This study tries to compare three backbone deep learning architectures namely, ResNet 50, MobileNetv3 Large and EfficientNet B3 on an image advertisement dataset to classify the underlying sentiments being perceived by the consumers. Transfer learning is used to mitigate the small dataset problem.

EfficientNet performed the best overall but the performance was still very poor. Grad-CAM visualizations confirmed our understanding of this model and helped us gain more confidence on the performance of the model.


## Directory Structure

```
├── documentation/              <- All project related documentation and reports
├── notebooks/                  <- Jupyter notebooks
│  ├── data_preprocessing/      <- Notebooks for cleaning the dataset
│  ├── image_preprocessing/     <- Notebooks for processing images and visualization
├── src/                        <- Source code for the project
│  ├── multilabel/              <- Scripts for the multilabel dataset
│  ├── __init__.py              <- Makes src a Python module
├── .gitignore                  <- List of files and folders git should ignore
├── LICENSE                     <- Project's License
├── README.md                   <- The top-level README for developers using this project
└── environment.yml             <- Conda environment file
```

## Creating the environment
Load conda environment as:
```
conda env create -f environment.yml
```
Install torch in conda environment:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Dataset Used

We have used a publicly available dataset developed by the combined efforts of Hussain et al. at the University of Pittsburgh with over 64,000 advertisement images and over 3,000 video advertisements. The authors used Amazon Mechanical Turk workers to tag each advertisement to its respective topic (eg. category of the product the advertisement targets) and what sentiment it conveys to the viewer (eg. how plants/trees play a vital role in sustenance) followed by what method it uses to imbibe that message (eg. the presence of trees or plants might be depicting life). The approach used to gather and annotate this data was influenced by the research in Media
Studies, an academic field that examines the content of mass media messages, with input from one of the research paper authors who had formal education in the field. The data is accessible at http://www.cs.pitt.edu/kovashka/ads/.

![dataset](/figures/dataset.png)

## Methodology

Before training, the images were analysed to come up with
a pre-processing pipeline to denoise the images and improve
their quality as most of the images were highly compressed.

<ol type="A">
 <li><b>Bilateral Filter</b></li>
For improving the quality of the images, a smoothing filter
for images had to be employed. So, a bilateral filter was used to reduce noise while preserving edges in a non-linear manner. It is quintessential to know that all other
filters smudge the edges, while Bilateral Filtering retains
them.

![ads-bilateral](/figures/ads-bilateral.png)

 <li><b>Pre-processing Techniques</b></li>
Pytorch provides various functional transformations that
can be applied using the torchvision.transform module. But, they require a parameter such as a factor by which an image can be transformed, therefore they cannot be applied to all images with the same factor. 

Five random images have been selected and functional image processing techniques like hue transforms, gamma transforms, solarize transformations, sharpness, etc have been applied to reach a conclusion that all images bearing uniqueness in their characteristics respond differently to functional transformations applied.

![pre-processing](/figures/pre-processing.png)
 
 <li><b>Architectures</b></li>
Different backbone architectures were chosen to ensure
that different types of Convolution blocks were tested for the advertisement data. <b>Resnet-50</b>, <b>MobileNet V3 Large</b> and <b>EfficientNet B3</b> were chosen finally.

| Architecture      | Params (Mil.) | Layers | GFLOPS | Imagenet Acc. |
|-------------------|---------------|--------|--------|---------------|
| MobileNet V3 Large|`5.5`          |`18`    | `0.22` | `92.57`       |
| EfficientNet B3   |`12.2`         |`29`    | `1.83` | `96.05`       |
| Resnet-50         |`25.6`         |`50`    | `4.09` | `95.43`       |
</ol>

## Results
The dataset used in this study presented the the multiclass, multilabel classification problem. Thus, to make the model predict multiple labels, a sigmoid layer had to be added before the loss function to get 0 or 1 prediction for all the classes of the data. To achieve this, the BCEWITHLOGITSLOSS function of PyTorch was used as it combines the Sigmoid layer and the binary cross entropy loss function in one single class. This makes theses operations more numerically stable than their separate counterparts.

The pre-trained weights were chosen to be the IMAGENET1K V2 weights and only the last classification layer was fine-tuned. The rationale behind performing this type
of shallow-tuning was that the Imagenet data is very similar to the advertisement images in our dataset. Additionally, the size of the selected dataset is small so deep-tuning might not work well.

![f1&loss_plots](/figures/f1&loss_plots.png)

| Model             | F1 Score | Time | F1 Epochs | Loss Epochs |
|-------------------|----------|------|-----------|-------------|
| MobileNet V3 Large|`0.168`   |`80s` | `50`      | `98`        |
| EfficientNet B3   |`0.189`   |`153s`| `5`       | `90`        |
| Resnet-50         |`0.179`   |`50s` | `10`      | `0`         |

The best model which in this case was the **EfficientNet B3** model was used to do further analysis like visualizing the trained filters and using Grad-CAM to understand which areas of the image the model focused on to generate the predictions.