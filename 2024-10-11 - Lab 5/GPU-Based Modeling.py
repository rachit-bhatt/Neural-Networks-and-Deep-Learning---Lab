import pandas as pd
import numpy as np
import cudf
from cuml import PCA as cuML_PCA
from cuml.linear_model import LogisticRegression as cuML_LogisticRegression
from cuml.ensemble import RandomForestClassifier as cuML_RandomForestClassifier
from cuml.svm import SVC as cuML_SVC
from cuml.neighbors import KNeighborsClassifier as cuML_KNNClassifier
from cuml.tree import DecisionTreeClassifier as cuML_DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import plotly.graph_objects as go

# Step 1: Data loading and preprocessing (on CPU)
# Load your data here (assuming it's stored in Pandas format for CPU)
# df = pd.read_csv("your_data.csv")

# Assuming you've already split your data into X (features) and y (target)
# X_train, X_test, y_train, y_test = ...

# Identify numeric and categorical columns
numeric_features = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Preprocessing pipeline for CPU (StandardScaler for numeric, OneHotEncoder for categorical)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Apply the preprocessing pipeline on the CPU
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Convert the preprocessed data back to a DataFrame (Pandas for CPU)
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed)
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed)

# Step 2: Convert the data to cuDF format for GPU-based processing
X_train_gpu = cudf.DataFrame.from_pandas(X_train_preprocessed_df)
X_test_gpu = cudf.DataFrame.from_pandas(X_test_preprocessed_df)
y_train_gpu = cudf.Series(y_train)
y_test_gpu = cudf.Series(y_test)

# Step 3: Dimensionality reduction with GPU-based PCA
pca_gpu = cuML_PCA(n_components=10)
X_train_reduced_gpu = pca_gpu.fit_transform(X_train_gpu)
X_test_reduced_gpu = pca_gpu.transform(X_test_gpu)

# Step 4: Train models using cuML (on GPU)

# Define models using cuML
models = {
    "Logistic Regression": cuML_LogisticRegression(),
    "Decision Tree": cuML_DecisionTreeClassifier(),
    "Random Forest": cuML_RandomForestClassifier(),
    "K-Nearest Neighbors": cuML_KNNClassifier(),
    "Support Vector Classifier": cuML_SVC(),
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_reduced_gpu, y_train_gpu)
    
    # Predict on test data
    y_pred_gpu = model.predict(X_test_reduced_gpu)
    
    # Convert predictions back to CPU for evaluation
    y_pred = y_pred_gpu.to_pandas()
    y_test_cpu = y_test_gpu.to_pandas()
    
    # Calculate accuracy and precision
    accuracy = accuracy_score(y_test_cpu, y_pred)
    precision = precision_score(y_test_cpu, y_pred, pos_label='>50K')
    
    # Save results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test_cpu, y_pred)

    # Plot confusion matrix with values inside
    def plot_confusion_matrix(cm, model_name):
        x_labels = ['<=50K', '>50K']  # Predicted labels
        y_labels = ['<=50K', '>50K']  # Actual labels

        annotations = []
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=y_labels[i],
                        text=str(cm[i][j]),
                        showarrow=False,
                        font=dict(color='black')
                    )
                )

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=x_labels,
            y=y_labels,
            colorscale='Blues',
            colorbar=dict(title='Count'),
            hoverongaps=False
        ))

        fig.update_layout(
            annotations=annotations,
            title=f'Confusion Matrix for {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            width=500
        )

        fig.show()

    # Plot confusion matrix
    plot_confusion_matrix(cm, name)

# Step 5: Model comparison
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print('-' * 30)