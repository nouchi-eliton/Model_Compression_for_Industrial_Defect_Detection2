# Model Compression for Industrial Defect Detection
In industrial quality control, defect detection is a crucial process that aims to identify defective products before they leave the production line. Deep learning models are a good option to this task; However, very deep architectures require powerful hardware for deployment and can introduce bottlenecks in the production line due to high inference latency. For these reasons, this project explores the application of model compression techniques to visual defect detection models, aiming to reduce models storage size and inference latency.

## Compression techniques
- **Knowledge distillation** consists of training a large, complex neural network (the teacher) to guide the training of a smaller model (the student). This technique enables the use of a faster and more compact models with minimal loss in accuracy.

- **Network quantization** is a technique that reduces the numerical precision of the network weights and activations while preserving similar performance.

## Dataset
This project uses the MvTecAD dataset to validate the methodology. The dataset contains 15 different industrial products that can be either normal or defective. There are several types of defects, ranging from stains and scratches to misplaced parts.

The dataset is public available in the folowing link: https://www.mvtec.com/company/research/datasets/mvtec-ad
