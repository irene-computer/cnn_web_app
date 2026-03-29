# CNN Vision - Image Classification with Convolutional Neural Networks

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-2.3.3-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Accuracy-78%25-success.svg" alt="Accuracy">
</div>

<br>

<p align="center">
  <strong>A professional web application for image classification using Convolutional Neural Networks (CNN) trained on the CIFAR-10 dataset.</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-documentation">API</a> •
  <a href="#model-architecture">Model</a> •
  <a href="#performance">Performance</a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

**CNN Vision** is a complete end-to-end deep learning project that implements a Convolutional Neural Network for image classification. The model is trained on the CIFAR-10 dataset, which contains **60,000** 32x32 color images across **10 different classes**. The application provides an intuitive web interface where users can upload images and get real-time predictions with confidence scores.

### Key Statistics
- **10 classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **78%+ accuracy** on test set
-  **814,122 trainable parameters**
- **Real-time predictions** (<100ms per image)
-  **RESTful API** for programmatic access

---

## Features

### Core Functionality
- **Image Upload**: Drag-and-drop or click to upload images
- **Multi-format Support**: PNG, JPG, JPEG, GIF, BMP
- **Top-5 Predictions**: Shows the 5 most likely classes with confidence percentages
- **Visual Feedback**: Displays the uploaded image with prediction results
- **Real-time Processing**: Instant classification results

### Testing Features
- **Random Testing**: Test the model with random CIFAR-10 images
- **Batch Testing**: Run batch tests with 10 random images to evaluate accuracy
- **Specific Index Testing**: Test specific images by their dataset index (0-9999)

### Analytics & Visualization
- **Training Curves**: Visualize loss and accuracy during training
- **Confusion Matrix**: Detailed classification metrics per class
- **Performance Metrics**: Best validation accuracy, loss, and epochs
- **Misclassified Examples**: Display images where the model made mistakes

### Web Interface
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Dark/Light Theme**: Professional gradient-based design
- **Interactive Elements**: Smooth animations and hover effects
- **User-friendly Navigation**: Intuitive menu structure

---

## Demo

### Home Page
![Home Page](https://via.placeholder.com/800x400?text=Home+Page+Screenshot)

### Prediction Interface
![Prediction Page](https://via.placeholder.com/800x400?text=Prediction+Page+Screenshot)

### Test Mode
![Test Page](https://via.placeholder.com/800x400?text=Test+Page+Screenshot)

### Performance Analytics
![Performance Page](https://via.placeholder.com/800x400?text=Performance+Page+Screenshot)

---

##  Project Structure
