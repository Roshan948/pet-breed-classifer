# Oxford-IIIT Pet Breed Classification — Project Summary

## 1. About the Dataset

For this project, I used the **Oxford-IIIT Pet Dataset**, which contains 7,349 images of cats and dogs belonging to 37 different breeds. There are 12 cat breeds and 25 dog breeds. Most breeds have around 200 images, although the exact number varies slightly between breeds.

The images are quite different from each other in terms of size, lighting, pose, and background. The dataset also provides breed labels, head bounding boxes, and segmentation information for the images.

## 2. Problem I Tried to Solve

The main goal of the project was to build an image classification model that can identify the breed of a cat or dog from an image.

This is more difficult than simply classifying an image as a cat or a dog because some breeds look very similar. For example, some terrier breeds have similar body shapes and coat patterns. So, the model needs to learn smaller features such as the shape, color, and texture of the animal.

Basically, model le photo herera breed guess garnu parcha, which is why this is considered a fine-grained classification problem.

## 3. Data Analysis and Preprocessing

Before training the models, I first checked the dataset to understand whether there were any problems with the images.

The EDA showed that there were no corrupted or missing images and no duplicate entries. Only 3 images were not in RGB format, so those images needed to be converted. The average image size was around 435 × 402 pixels, so resizing was necessary before giving the images to the CNN models.

I resized all images to **160 × 160 pixels**.

For the custom CNN, pixel values were scaled between **0 and 1**. For MobileNetV2, I used its own `preprocess_input` function, which scales the input to the range expected by the pretrained model.

I also used data augmentation on the training images. This included:

* Random horizontal flipping
* Small rotations of up to ±15°
* Random zoom
* Contrast changes

The main reason for doing this was to give the model slightly different versions of the same images. Since each breed has only around 200 images, this helped reduce the chance of the model simply memorizing the training images.

## 4. Train, Validation and Test Split

For splitting the dataset, I used the official `trainval.txt` and `test.txt` files provided with the dataset.

From the training/validation portion, I further created:

* **Training:** 3,128 images
* **Validation:** 552 images
* **Official Test:** 3,669 images

The test set was kept separate and was not used during training.

I also used a stratified split so that the different breeds remained reasonably balanced in the training and validation sets. This was important because I wanted to make sure that one breed was not overrepresented in one split.

## 5. Models Used

I experimented with two different approaches.

### Custom CNN

First, I created a CNN from scratch. It had three main convolution blocks:

`Conv2D → Batch Normalization → ReLU → MaxPooling`

The blocks used 32, 64 and 128 filters respectively.

After the convolution layers, I used:

* Global Average Pooling
* Dropout (0.4)
* 37-class Softmax output

The idea was to have a simple CNN as a baseline and then compare its performance with transfer learning.

### MobileNetV2

The second approach used **MobileNetV2**, which was already pretrained on ImageNet.

Initially, I kept the pretrained layers frozen and added a new classification head for the 37 pet breeds. After that, I fine-tuned the last approximately 30 layers using a much smaller learning rate of `1e-5`.

This approach made sense because the dataset is relatively small. Training a complete deep network from the beginning would require much more data. MobileNetV2 already knows how to detect many basic visual features, so I could reuse those features for this project.

## 6. Why I Used MobileNetV2

One reason I selected MobileNetV2 was that it is relatively lightweight compared with larger CNN architectures.

Most of my training was done without GPU access, so using a smaller model was more practical. It also gave me a good opportunity to see the actual difference between a CNN trained from scratch and a pretrained model.

Honestly, custom CNN bata suru garda result ekdam low aayo, so transfer learning ko benefit clearly dekhiyo.

The custom CNN was mainly useful as a baseline. It helped me understand how much pretrained features can improve performance when the dataset is not very large.

## 7. Experiments and Results

I tried different settings for the custom CNN and then compared them with MobileNetV2.

| Model                       | Validation Accuracy | Epochs |
| --------------------------- | ------------------: | -----: |
| Custom CNN – Adam, LR 1e-3  |              10.69% |     20 |
| Custom CNN – SGD + Momentum |               6.34% |      8 |
| Custom CNN – LR 1e-4        |              10.33% |      8 |
| MobileNetV2 – Frozen        |              90.04% |     10 |
| MobileNetV2 – Fine-tuned    |          **90.22%** |      6 |

The difference between the two approaches was quite large. The custom CNN stayed around 10–11% validation accuracy, while MobileNetV2 reached around 90%.

So, in this particular project, transfer learning worked much better than training the CNN completely from scratch.

## 8. Final Test Result

After selecting the fine-tuned MobileNetV2 model, I evaluated it on the test set that had not been used during training.

The final results were:

* **Test Accuracy:** 86.59%
* **Test Loss:** 0.4201
* **Macro Precision:** 0.8752
* **Macro Recall:** 0.8658
* **Macro F1 Score:** 0.8650

The test accuracy was lower than the validation accuracy, which is expected because the test images were completely held out from the training process.

## 9. What I Observed From the Results

The biggest thing I noticed was the difference between the custom CNN and MobileNetV2.

The custom CNN only reached around 10–11% validation accuracy, while MobileNetV2 reached about 90%. With 37 classes, random guessing would be around 2.7%, so the custom CNN was learning something, but it was still not good enough for practical classification.

MobileNetV2, on the other hand, performed much better because it already had useful visual features learned from ImageNet.

Fine-tuning improved the validation accuracy slightly from **90.04% to 90.22%**. The improvement was not huge, but it showed that allowing some of the pretrained layers to adjust to the pet dataset was useful.

## 10. Strong and Weak Classes

Some breeds were classified very well. The classes with F1 scores of 0.95 or above included:

* Keeshond — 0.98
* Great Pyrenees — 0.97
* Pomeranian — 0.96
* Samoyed — 0.96
* Shiba Inu — 0.96
* Pug — 0.95
* Yorkshire Terrier — 0.95

These breeds generally have more noticeable features such as their coat, color, or body shape.

Some breeds were more difficult for the model.

The **American Pit Bull Terrier** had a recall of 0.37, while the **Staffordshire Bull Terrier** had a precision of 0.55. These two breeds were one of the major confusion pairs because they have fairly similar body shapes and short coats.

The model also struggled with **Ragdoll**, which had a precision of 0.60 and could be confused with visually similar cat breeds such as Birman and Persian.

## 11. Validation vs Test Accuracy

The validation accuracy was about **90.2%**, while the final test accuracy was about **86.6%**.

There is therefore a difference of roughly 4 percentage points between them. For this dataset, this is a relatively modest generalization gap and does not by itself suggest serious overfitting.

## 12. Final Application

The complete model pipeline was implemented using **TensorFlow/Keras**.

The trained model and preprocessing configuration were saved in the `saved_model/` folder.

I also created a simple **Streamlit application** called `pet_breed_app.py`. The user can upload a pet image through the application, and the model predicts the breed along with its confidence and the top-3 possible predictions.

So, overall, the project was not only about training a model but also about putting the trained model into a small application that can actually be used for prediction.
