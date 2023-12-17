# ElasticTrain
A flexible compressed DNN training framework, capable of training robust DNNs while simultaneously satisfying a mixture of different constraints, including unstructured, structured, pattern-based, and a novel workload-balanced pruning, as well as multi-codebook quantization.

# Updates
We will soon submit a research paper under the title of "ElasticTrain: A Flexible, Ultra-High Compression DNN Training Framework", and we will open-source the framework after publication; however, some of our ultra-compressed models with the starter code are available here at the moment.

# Sample Models
The table below lists some of our trained models available here early, and summarises their compression details and accuracy.
All the listed models can be found in `TrainedModels` directory.

| Dataset | Model | Overall Compression | Pruning Rtio | Pruning Structure | Quantization Bits / Compression | Codebooks per Layer | Accuracy |
| ------- | ----- | ------------------- | ------------ | ----------------- | ------------------------------- | ------------------- | -------- |
| CIFAR10 | VGG16 |         50×         |     50×      |    unstructured   |               -                 |          -          |  93.26%  |
| CIFAR10 | VGG16 |         500×        |     500×     |    unstructured   |               -                 |          -          |  91.17%  |
<hr style="height:2px;border-width:0;color:gray;background-color:gray">
| CIFAR10 | VGG16 |         100×        |     100×     |       kernel      |               -                 |          -          |  92.25%  |
| CIFAR10 | VGG16 |         500×        |     500×     |       kernel      |               -                 |          -          |  88.59%  |
|         |       |                     |              |                   |                                 |                     |          |
| CIFAR10 | VGG16 |         8000×       |     1000×    |    unstructured   |              4/8×               |          1          |  88.59%  |
| CIFAR10 | VGG16 |         16000×      |     1000×    |    unstructured   |              2/16×              |          64         |  87.93%  |

# Testing the Models
A simple starter code is provided in `ModelEval.py` to assist with loading, testing, and exploring the compressed trained models.
