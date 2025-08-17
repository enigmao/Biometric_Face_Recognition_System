
# Facial Recognition Authentication (Siamese Network)

This project implements a **Facial Recognition Authentication System** using a **Siamese Network**.  
It uniquely identifies individuals during authentication activities like login by learning similarity functions.

- **Model**: Siamese Network (TensorFlow/Keras)  
- **Dataset**: [LFW - Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)  
- **Reference Paper**: [Koch et al. - Siamese Neural Networks for One-Shot Learning](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)  

## Installation
```bash
pip install -r requirements.txt
```

## Run Locally
```bash
python application.py
# open http://localhost:5000
```

## Deploy on Heroku/Render
- Ensure `Procfile` and `requirements.txt` are included.
- Add your trained `siamese_model.h5` at the project root.
- Deploy 🚀

## Usage
- Upload two face images.
- The system outputs a similarity score between 0 and 1.
- Higher scores mean higher similarity.
