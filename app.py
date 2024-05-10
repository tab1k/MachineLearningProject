import pandas as pd
from preprocessing import preprocess_data
from model_evaluation import decision_tree_accuracy, kmeans_accuracy, svm_accuracy
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


class MachineLearningApp:
    def __init__(self):
        self.file_path = "data/dataset.csv"
        self.df = pd.read_csv(self.file_path)
        self.preprocessed_df = preprocess_data(self.df)

    def load_data(self):
        return self.df

    def preprocess_data(self):
        return self.preprocessed_df

    def display_model_performance(self):
        st.header("Model Performance")
        st.write("Accuracy of Decision Tree Classifier:", decision_tree_accuracy)
        st.write("Accuracy of KMeans:", kmeans_accuracy)
        st.write("Accuracy of Support Vector Machine:", svm_accuracy)
        print(self.preprocessed_df.columns)
        print(self.df.columns)

    def visualize_feature(self, selected_feature):
        if selected_feature == "Indicate your gender":
            plt.figure()
            sns.countplot(data=self.preprocessed_df, x="Indicate your gender")
            plt.title("Distribution of Indicate your gender")
            plt.xlabel("Gender")
            plt.ylabel("Count")
            st.pyplot(plt)
        elif selected_feature == "Please indicate your age":
            plt.figure()
            sns.histplot(data=self.preprocessed_df, x="Please indicate your age", bins=10)
            plt.title("Distribution of Age")
            plt.xlabel("Age")
            plt.ylabel("Count")
            st.pyplot(plt)

    def display_visualizations(self):
        st.header("Visualizations")
        selected_feature = st.selectbox("Select feature for visualization:", self.preprocessed_df.columns)
        self.visualize_feature(selected_feature)

        fig, ax = plt.subplots()
        ax.bar(["Decision Tree", "KMeans", "Support Vector Machine"],
               [decision_tree_accuracy, kmeans_accuracy, svm_accuracy])
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy")
        st.pyplot(fig)

        if "Indicate your gender" in self.preprocessed_df.columns:
            plt.figure()
            sns.countplot(data=self.preprocessed_df, x="Indicate your gender")
            plt.title("Distribution of Indicate your gender")
            plt.xlabel("Gender")
            plt.ylabel("Count")
            st.pyplot(plt)

        if "Please indicate your age" in self.df.columns:
            plt.figure()
            sns.histplot(data=self.df, x="Please indicate your age", bins=10)
            plt.title("Distribution of Age")
            plt.xlabel("Age")
            plt.ylabel("Count")
            st.pyplot(plt)

        if "Which form of domestic violence is most prevalent?" in self.preprocessed_df.columns:
            plt.figure()
            sns.countplot(data=self.preprocessed_df, x="Which form of domestic violence is most prevalent?")
            plt.title("Distribution of Forms of Domestic Violence")
            plt.xlabel("Form of Violence")
            plt.ylabel("Count")
            st.pyplot(plt)

        if "Which organization provides assistance to domestic violence victims?" in self.preprocessed_df.columns:
            plt.figure()
            sns.countplot(data=self.preprocessed_df, x="Which organization provides assistance to domestic violence victims?")
            plt.title("Distribution of Assistance Organizations")
            plt.xlabel("Organization")
            plt.ylabel("Count")
            st.pyplot(plt)

        if "Which professionals can assist domestic violence victims?" in self.preprocessed_df.columns:
            plt.figure()
            sns.countplot(data=self.preprocessed_df, x="Which professionals can assist domestic violence victims?")
            plt.title("Distribution of Professionals Helping Domestic Violence Victims")
            plt.xlabel("Professionals")
            plt.ylabel("Count")
            st.pyplot(plt)

        correlation_matrix = self.preprocessed_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        st.pyplot(plt.gcf())

        sns.lmplot(x="Indicate your gender", y="Please indicate your age", data=self.preprocessed_df)
        plt.title("Regression Plot")
        plt.xlabel("Gender")
        plt.ylabel("Age")
        st.pyplot(plt.gcf())


def main():
    st.title("Machine Learning Web Application")
    app = MachineLearningApp()
    app.display_model_performance()
    app.display_visualizations()


if __name__ == "__main__":
    main()
