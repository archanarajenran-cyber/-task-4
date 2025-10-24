"""
Breast Cancer Classification using Logistic Regression
Task 4: Binary Classification with Evaluation Metrics
Author: AI & ML Internship Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, precision_score, recall_score, 
                             accuracy_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BreastCancerClassifier:
    """
    A comprehensive Logistic Regression classifier for Breast Cancer detection
    """
    
    def __init__(self, data_path='data.csv'):
        """Initialize the classifier with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def load_and_explore_data(self):
        """Load and explore the dataset"""
        print("=" * 80)
        print("STEP 1: LOADING AND EXPLORING DATA")
        print("=" * 80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of samples: {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1]}")
        
        # Check for missing values
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        # Display class distribution
        print("\nClass Distribution:")
        print(self.df['diagnosis'].value_counts())
        print("\nClass Percentages:")
        print(self.df['diagnosis'].value_counts(normalize=True) * 100)
        
        # Display first few rows
        print("\nFirst 5 rows of the dataset:")
        print(self.df.head())
        
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data: encode labels, split, and standardize"""
        print("\n" + "=" * 80)
        print("STEP 2: PREPROCESSING DATA")
        print("=" * 80)
        
        # Convert diagnosis to binary (M=1, B=0)
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0})
        
        # Separate features and target
        X = self.df.drop(['id', 'diagnosis'], axis=1)
        y = self.df['diagnosis']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")
        
        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("\n✓ Features standardized (mean=0, std=1)")
        
    def train_model(self, max_iter=10000):
        """Train the Logistic Regression model"""
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING LOGISTIC REGRESSION MODEL")
        print("=" * 80)
        
        self.model = LogisticRegression(max_iter=max_iter, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        print("\n✓ Model trained successfully!")
        print(f"Model coefficients shape: {self.model.coef_.shape}")
        print(f"Model intercept: {self.model.intercept_[0]:.4f}")
        
    def make_predictions(self, threshold=0.5):
        """Make predictions on test set"""
        print("\n" + "=" * 80)
        print("STEP 4: MAKING PREDICTIONS")
        print("=" * 80)
        
        # Predict probabilities
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Apply threshold
        self.y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        print(f"\nPredictions made with threshold: {threshold}")
        print(f"Predicted positive cases (Malignant): {sum(self.y_pred)}")
        print(f"Predicted negative cases (Benign): {len(self.y_pred) - sum(self.y_pred)}")
        
    def evaluate_model(self):
        """Evaluate the model using various metrics"""
        print("\n" + "=" * 80)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 80)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print("\n📊 PERFORMANCE METRICS:")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Classification report
        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, self.y_pred, 
                                    target_names=['Benign (0)', 'Malignant (1)']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\n🔍 CONFUSION MATRIX:")
        print(f"True Negatives:  {cm[0, 0]}")
        print(f"False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]}")
        print(f"True Positives:  {cm[1, 1]}")
        
        return accuracy, precision, recall, roc_auc, cm
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix\nBreast Cancer Classification', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve\nBreast Cancer Classification', 
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved as 'roc_curve.png'")
        plt.show()
        
    def plot_sigmoid_function(self):
        """Plot the sigmoid function"""
        z = np.linspace(-10, 10, 100)
        sigmoid = 1 / (1 + np.exp(-z))
        
        plt.figure(figsize=(10, 6))
        plt.plot(z, sigmoid, 'b-', linewidth=2, label='Sigmoid Function')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
        plt.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z = 0')
        plt.xlabel('z (Linear Combination)', fontsize=12)
        plt.ylabel('σ(z) - Probability', fontsize=12)
        plt.title('Sigmoid Activation Function\nσ(z) = 1 / (1 + e^(-z))', 
                  fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('sigmoid_function.png', dpi=300, bbox_inches='tight')
        print("✓ Sigmoid function plot saved as 'sigmoid_function.png'")
        plt.show()
        
    def tune_threshold(self):
        """Experiment with different threshold values"""
        print("\n" + "=" * 80)
        print("STEP 6: THRESHOLD TUNING")
        print("=" * 80)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        results = []
        
        print("\nTesting different threshold values:")
        print("-" * 60)
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            precision = precision_score(self.y_test, y_pred_thresh)
            recall = recall_score(self.y_test, y_pred_thresh)
            accuracy = accuracy_score(self.y_test, y_pred_thresh)
            
            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall
            })
            
            print(f"Threshold {threshold:.1f} | "
                  f"Accuracy: {accuracy:.4f} | "
                  f"Precision: {precision:.4f} | "
                  f"Recall: {recall:.4f}")
        
        return pd.DataFrame(results)
    
    def explain_concepts(self):
        """Explain key concepts"""
        print("\n" + "=" * 80)
        print("CONCEPT EXPLANATIONS")
        print("=" * 80)
        
        print("\n1️⃣ SIGMOID FUNCTION:")
        print("   The sigmoid function σ(z) = 1/(1+e^(-z)) maps any real-valued number")
        print("   to a value between 0 and 1, representing probability.")
        print("   It's used in logistic regression to convert linear predictions")
        print("   to probabilities for binary classification.")
        
        print("\n2️⃣ PRECISION vs RECALL:")
        print("   • Precision = TP/(TP+FP) - Of all positive predictions, how many are correct?")
        print("   • Recall = TP/(TP+FN) - Of all actual positives, how many did we catch?")
        print("   • High Precision: Few false alarms")
        print("   • High Recall: Few missed cases")
        
        print("\n3️⃣ ROC-AUC CURVE:")
        print("   • ROC (Receiver Operating Characteristic) plots TPR vs FPR")
        print("   • AUC (Area Under Curve) measures overall model performance")
        print("   • AUC = 1.0: Perfect classifier")
        print("   • AUC = 0.5: Random classifier")
        print("   • Higher AUC = Better model performance")
        
        print("\n4️⃣ CONFUSION MATRIX:")
        print("   A table showing correct and incorrect predictions:")
        print("   • True Positives (TP): Correctly predicted malignant")
        print("   • True Negatives (TN): Correctly predicted benign")
        print("   • False Positives (FP): Wrongly predicted malignant (Type I error)")
        print("   • False Negatives (FN): Wrongly predicted benign (Type II error)")
        

def main():
    """Main execution function"""
    print("🏥 BREAST CANCER CLASSIFICATION PROJECT 🏥")
    print("Using Logistic Regression for Binary Classification\n")
    
    # Initialize classifier
    classifier = BreastCancerClassifier('data.csv')
    
    # Execute pipeline
    classifier.load_and_explore_data()
    classifier.preprocess_data()
    classifier.train_model()
    classifier.make_predictions()
    accuracy, precision, recall, roc_auc, cm = classifier.evaluate_model()
    
    # Visualizations
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    classifier.plot_confusion_matrix(cm)
    classifier.plot_roc_curve()
    classifier.plot_sigmoid_function()
    
    # Threshold tuning
    threshold_results = classifier.tune_threshold()
    
    # Explain concepts
    classifier.explain_concepts()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n🎯 Final Model Performance:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print("\n📁 Generated Files:")
    print("   • confusion_matrix.png")
    print("   • roc_curve.png")
    print("   • sigmoid_function.png")
    print("\n💡 Next Steps:")
    print("   • Upload this code to GitHub")
    print("   • Include the generated visualizations")
    print("   • Add a comprehensive README.md")
    print("   • Document your findings and insights")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()