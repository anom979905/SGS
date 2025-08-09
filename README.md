♻ Smart Garbage Segregation – AI-Powered Waste Management
Overview
This project focuses on using AI and Machine Learning to automate the process of identifying and sorting waste. Through image classification, the system can recognize materials like plastic, paper, and metal, and assign them to the correct recycling category. By training a model with a large dataset of labeled waste images, the system can predict the category of new, unlabeled waste efficiently and accurately.

The goal is to improve recycling efficiency, reduce human error, and recover more recyclable materials by replacing slow, labor-intensive manual sorting with AI-based automation.

Inspiration
In 2021, the world generated over 2.01 billion tons of municipal solid waste.

33% of that waste wasn’t managed safely.

Around 8 million metric tons of plastic reach oceans annually — that’s like five grocery bags of plastic for every foot of shoreline.

Manual garbage segregation is often slow and inefficient. This project uses computer vision and machine learning to detect and classify waste instantly, improving efficiency while reducing environmental pollution. The system can self-learn from mistakes, becoming more accurate over time.

Social Impact
Implementing AI-based waste classification brings:

Environmental Sustainability – Increases recyclable recovery, reduces waste, and lowers greenhouse gas emissions.

Job Creation – Opens opportunities for skilled roles in technology, AI, and recycling process management.

Awareness & Education – Promotes recycling awareness through improved processes and visible impact.

Equity & Access – Makes recycling more accessible in underserved communities through efficiency gains.

Tech Stack
Intel oneAPI – Cross-architecture optimization tools for CPUs, GPUs, and FPGAs.

OneDNN – Optimized deep learning operations (convolutions, pooling, normalization, activation).

Python – Core programming language.

TensorFlow – Deep learning framework for CNN-based image classification.

Jupyter Notebook – Development and experimentation environment.

Streamlit – Deployment for interactive web-based access.

Intel oneAPI & OneDNN Optimization
By enabling:

python
Copy
Edit
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
TensorFlow automatically uses OneDNN optimizations for faster execution of CNN layers like Conv2D and Dense. This results in quicker training and inference times on compatible Intel hardware.

What It Does
Captures waste images via a camera.

Uses a trained CNN model to classify waste into categories.

Automates sorting to reduce manual effort, boost efficiency, and minimize pollution.

Encourages better waste disposal habits by making users aware of the types of waste they generate.

How We Built It
Import libraries

Load dataset

Explore and preprocess data

Prepare training/testing generators

Save category labels to Labels.txt

Design CNN model (3 conv blocks, pooling, dropout layers)

Compile model (Adam optimizer, sparse categorical crossentropy loss, accuracy metric)

Train – batch size: 32, epochs: 10

Test predictions

Save model as modelnew.h5

Deploy with Streamlit for real-time use

Key Learnings
OneDNN Optimization – Boosted speed and efficiency of CNN training and inference.

Waste Management – Gained insights into eco-friendly disposal, recycling, and sustainability practices.

Image Processing – Mastered preprocessing, feature extraction, and enhancement techniques.

Machine Learning – Built and trained CNN models for image classification.

Model Evaluation – Applied metrics like accuracy, precision, and recall to refine results.

Data Analysis – Improved skills in cleaning, wrangling, and visualizing datasets.

Sustainability Awareness – Learned the importance of reducing waste and promoting recycling.

Collaboration – Experienced teamwork across domains like waste management, AI, and analytics.

✅ This project not only solves a real-world environmental problem but also develops valuable skills in AI, ML, optimization, and sustainability — making it a powerful step toward smarter waste management.
