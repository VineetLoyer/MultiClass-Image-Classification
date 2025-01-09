
This repository contains a **multiclass image classification project** implemented using deep learning models like **VGG16**, **ResNet50**,**ResNet101**, and **EfficientNetB0**. The project is developed using **TensorFlow** and **Keras** for training, evaluation, and performance comparison.

---

## **Overview**

- **Objective**: Classify images into multiple categories.
- **Dataset**: A labeled dataset with the following categories:
  - Buildings
  - Forest
  - Glacier
  - Mountain
  - Sea
  - Street

- **Models Implemented**:
  - VGG16
  - ResNet50
  - ResNet101
  - EfficientNetB0

- **Tasks**:
  1. Use pretrained models ResNet50, ResNet100, EfficientNetB0, and VGG16
  2. Only train the last fully connected layer and freeze all layers before them.
  3. Use the output of penultimate layer as features extracted from each image.
  4. Use ReLU activation functions in the last layer, softmax layer, L2 regularization, batch normalizaton, dropoutrate of 20% and ADAM optimizer.
  5. Can try any batch size.
  6. Train the model for atleast 50 epochs and perform early stopping using validation set.(validation set comprise of random set of 20% of each class)
  7. Keep the network parameters that have lowest validation error. Plot training and validation error vs epochs.
  8. Report training, validation and test Precision, Recall, AUC, and F1 scorefor those models. And provide comparison.

---

## **Project Structure**

```plaintext
final-project-VineetLoyer/
│
├── notebook/
│   ├── Loyer_Vineet_final_project_DSCI552.ipynb           # Jupyter Notebook 
│
├── data/
│   ├── seg_train/                    # Training images
│   ├── seg_test/                     # Test images
│
└── README.md                     # Project documentation


```
- **Libraries and Dependencies**
    - Python 3.8+
    - TensorFlow 2.8+
    - Keras
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - OpenCV

---
## **Project Findings**

**ResNet50 performance on Test data**<br>
![image](https://github.com/user-attachments/assets/3ecf5c7b-5e3b-4a74-9da1-640fd7ce50ca)

**ResNet101 performance on Test data**<br>
![image](https://github.com/user-attachments/assets/84a21e7d-e569-43fe-bed3-7325446e991e)

**EfficientNetB0 performance on Test data**<br>
![image](https://github.com/user-attachments/assets/c79de483-d572-4ff5-a353-af10c3408c64)

**VGG16 performance on Test data**<br>
![image](https://github.com/user-attachments/assets/b2079155-c04c-4c2d-8b08-635b9a926912)



---
## **Conclusion**

  1. ResNet50 is the best-performing model overall, providing the highest balance of AUC, precision, recall, and F1-score.
  2. ResNet101 offers marginal improvements for some classes but is computationally more expensive, have comparable performance with 
     ResNet50.
  3. EfficientNetB0 is nearly as good as ResNet50 but with much lower computational cost,( takes lesser epochs to reach similar 
     accuracy).
  4. VGG16 underperforms and struggles with overlapping classes (eg: mountain and glacier).

---
