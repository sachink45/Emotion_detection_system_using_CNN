# Emotion_detection_system_using_CNN!

<img src="img2.png" alt="Result :" width="70%">

A CNN model is trained with grayscale images from the FER 2013 dataset to classify expressions into Seven emotions, namely happy, sad, neutral, disgust, surprise, fear and angry. To improve the accuracy and avoid overfitting of the model, batch normalization and dropout are used. It is a standard dataset which is available on kaggle.
The dataset link is : [click here](https://www.kaggle.com/datasets/msambare/fer2013)

# Softwares that required to build emotion_detection_system are :
- Python
- VS code
- tensorflow
- Keras
- Computer Vision (CV2)


# How to run?
### STEP 01:

Clone the repository

```bash
https://github.com/sachink45/Emotion_detection_system_using_CNN.git
```
### STEP 02- Create a conda environment after opening the repository
```bash
conda create --name <provide env_name>
```

### STEP 03- activate the conda env
```bash
conda activate env_name
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05- run evalution.py on the terminal by providing the input to - cap = cv2.VideoCapture('Please provide input here')
```bash
py evalution.py
```

### About Author
```bash
Author: Sachin Kapase
Data Scientist
Email: sachinkapase6125@gmail.com

```

