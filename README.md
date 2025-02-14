# 😷 Face Mask Detection and Gender Prediction

## 📁 Datasets
The datasets used for this project are available at this [Google Drive link](https://drive.google.com/drive/folders/1H75Z9RLkOxahyk8-4lN2HYE-a4_hSy8s?usp=sharing).

- **face_only_images**: Headshot photos of people wearing masks, categorized as Male or Female.
- **face-mask-detection-final**: Headshot photos of people categorized as WithMask or WithoutMask.
- **gender_prediction**: Headshot photos of people categorized as Male or Female.

The first and third datasets are subsets of a Kaggle dataset: [Medical Masks Part1](https://www.kaggle.com/datasets/tapakah68/medical-masks-part1/code?datasetId=1409004&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false).

The second dataset is a subset of [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset).

## 🚀 Approach

### 🛠️ Pre-Processing
The original Kaggle dataset for the third and first dataset was huge, about 80GB consisting of 40k images. I only used about 3000 images (evenly distributed into the two classes) to train my model. Furthermore, the images were selfies of people and not zoomed-in cropped photos of only their faces.

I had two options: I could use PIL (Python Imaging Library) or MTCNN (Multi-task Cascaded Convolutional Networks) for face detection as these are well-known for their results. I ran these for the first 10 images of the dataset and realized that MTCNN had much better accuracy in finding faces. PIL couldn't detect faces in some images, but MTCNN did.

So I used MTCNN to detect the face in the image and crop it out. A major difficulty I faced while applying MTCNN on the dataset was that the images were large as well, about 2000*1000 pixels. This required 10 hours to convert each image to its cropped version. Since most CNN image classification models take their image input as of the order 300 * 300, I decided to resize each image to 600 * 600 before applying MTCNN.

For the second dataset, luckily, I found a dataset that already contained the cropped face and small-sized version of the image, so I only had to reduce the size of the dataset. Then all the images were split into training, test, and validation sub-folders. Here, I am reducing the size of the dataset since even at about 3000 images (2000 train, 500 test, and 500 validation), the models which I had trained from scratch took about 150 minutes each to train.

### 🧠 Model Training and Testing
I planned and trained three models:
1. **Face Mask Detection**: Detects if a face mask is present.
2. **Gender Prediction with Face Mask**: Predicts gender for images with a person wearing a face mask.
3. **Gender Prediction without Face Mask**: Predicts gender for images without a face mask.

Even though the second model could theoretically work for the no-mask case, I wanted a higher accuracy and hence trained a separate model for it.

I used EfficientNetV2S as my base model and trained its weights from scratch on each particular task. I also added certain layers at the end so that the model can be used for binary classification using the sigmoid function. After training each model, I saved it in the `EfficientNetV2S` folder.

![image](https://github.com/vedh18/Task-3/assets/147409775/87360cbf-01f8-465b-8265-6eed85b4cbe1)
![image](https://github.com/vedh18/Task-3/assets/147409775/dfb4bbcd-a90d-4f6d-88df-ee9b3027c47b)
![image](https://github.com/vedh18/Task-3/assets/147409775/1c954e0b-8449-4e98-bc6a-a38dafaac325)

I then tested my model on the test data and plotted the confusion matrix to visualize the results.
![image](https://github.com/vedh18/Task-3/assets/147409775/058c44cf-c44d-4f03-a568-738fbad579a9)
![image](https://github.com/vedh18/Task-3/assets/147409775/f12ac49d-6072-4563-bc14-b16887310bd6)
![image](https://github.com/vedh18/Task-3/assets/147409775/d29c28a2-6666-4d78-96cc-88351c17f269)

### 💻 Final Implementation
To implement these models on any image and predict its gender, I again used MTCNN to crop out the face of the image and then applied these models using the logic stated before to get my results.
![image](https://github.com/vedh18/Task-3/assets/147409775/64113b2b-067f-4142-b232-2cfafb179a68)
![image](https://github.com/vedh18/Task-3/assets/147409775/b1e6db3f-2f44-45bd-a999-b4ff2babfa1e)
![image](https://github.com/vedh18/Task-3/assets/147409775/c76e6d3b-30ad-4b19-b163-34cc8b000477)
![image](https://github.com/vedh18/Task-3/assets/147409775/f6d245a9-d6ef-4c1f-860a-d7df128a4b2f)

## 🏃‍♂️ To Run the Code
Download the entire repository. Download the datasets from the [Google Drive link](https://drive.google.com/drive/folders/1H75Z9RLkOxahyk8-4lN2HYE-a4_hSy8s?usp=sharing) as it is and then simply run the `code.ipynb` file. To avoid training the model again, you could run the `run_model.ipynb` file to load models from the `EfficientNetV2S` folder directly.

---

Thank you for exploring this project! If you have any questions or suggestions, feel free to reach out.

📧 Email: vedhanthb23@iitk.ac.in
