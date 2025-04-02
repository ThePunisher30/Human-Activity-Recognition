# Human Activity Recognition

This project utilizes machine learning techniques to classify different human activities based on sensor data from the UCI HAR Dataset. Multiple machine learning models are implemented and evaluated for performance comparison.

## Features

- **Data Preprocessing**: Handling missing values, normalizing data, and preparing it for modeling.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and identifying key features.
- **Multiple Model Training**: Implementing various classification algorithms (Logistic Regression, SVM, Random Forest, KNN, etc.).
- **Model Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrices.
- **Visualization**: Comparing model accuracies and performance metrics through bar graphs and heatmaps.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Environment**: Jupyter Notebook / Python Script

## Dataset

The dataset used is the [UCI Human Activity Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). It contains sensor data collected from smartphones worn by 30 individuals while performing various activities.

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/ThePunisher30/Human-Activity-Recognition.git
```

### Navigate to the Project Directory

```bash
cd Human-Activity-Recognition
```

### Install Required Libraries

Ensure that you have Python installed. Install the necessary libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Script

If using Jupyter Notebook:

```bash
jupyter notebook
```

Open `HumanActivityRecognition.ipynb` and execute the cells to follow the analysis and model development.

If using a Python script:

```bash
python activity_recognition.py
```

## Results

The project trains multiple machine learning models and evaluates them using accuracy, classification reports, and confusion matrices. The models compared are:

- **Logistic Regression**
- **Linear Regression (for comparison purposes)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

Performance metrics are visualized using bar plots and heatmaps for better interpretability.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---

*Note: This README is based on the available information from the repository and may require additional details for complete setup and usage instructions.*
