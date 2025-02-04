import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from scipy import stats

class SurveyGramReport():
    def __init__(self, df, image_folder='images', pdf_name='final_report.pdf'):
        self.df = df.dropna()
        self.image_folder = image_folder
        self.pdf_name = pdf_name

    def profile_data(self):
        profile = ProfileReport(self.df, title="Profiling Report")
        profile.to_file(output_file='profiling_report.html')
        if os.path.exists('profiling_report.html'):
            print("Profiling Report Generated")
        else:
            raise FileNotFoundError("Profiling Report was not generated.")

    def create_visualizations(self):
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        
        sns.pairplot(self.df)
        plt.savefig(f"{self.image_folder}/pairplot.png")
        sns.pairplot(self.df, diag_kind='kde')
        plt.savefig(f"{self.image_folder}/pairplot_kde.png")

        for col in ['performance', 'km']:
            plt.scatter(self.df[col], self.df['price'])
            plt.xlabel(col)
            plt.ylabel("price")
            plt.title(f"KNN Prediction with Scatter Plot: /n{col} vs. price")
            plt.savefig(f"{self.image_folder}/{col}_scatter.png")
            plt.clf()
        data = self.df
        if 'performance' in data.columns:
            for col in ['km', 'performance']:
                print(col)
                plt.plot(self.df[col], self.df["price"])
                plt.xlabel(col)
                plt.ylabel("price")
                plt.title(f"KNN Prediction with Line Plot: {col} vs. price")
                plt.savefig(f"{self.image_folder}/{col}_line.png")
                plt.clf()
        print("Visualizations Created")

    def preprocess_data(self, test_size,random_state ):
        X = self.df.drop('price', axis=1)
        y = self.df['price']
        categorical_features = ["full_name", "gender", "model", "color", "condition"]
        transformer = ColumnTransformer([("one_hot", OneHotEncoder(), categorical_features)], remainder="passthrough")
        X_transformed = transformer.fit_transform(X)
        scaler = StandardScaler(with_mean=False)
        X_transformed = scaler.fit_transform(X_transformed)

        return train_test_split(X_transformed, y, test_size=test_size, random_state=random_state)

    def linear_regression(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Create a directory to store the plots
        plot_folder = os.path.join(self.image_folder, 'linear_regression_plots')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        # 1. Actual vs. Predicted Plot
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs. Predicted Prices")
        plt.savefig(os.path.join(plot_folder,'actual_vs_predicted.png'))
        plt.clf()  # Clear the figure

        # 2. Residuals vs. Fitted Plot
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Fitted Values")
        plt.axhline(y=0, color='r', linestyle='-')  # Add a horizontal line at 0
        plt.savefig(os.path.join(plot_folder, 'residuals_vs_fitted.png'))
        plt.clf()

        # 3. Normal Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Normal Q-Q Plot")
        plt.savefig(os.path.join(plot_folder, 'normal_qq_plot.png'))
        plt.clf()
        return y_pred, model

    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print("Mean Squared Error:", mse)
        print("R-squared:", r2)
        return mse, r2

    def create_report_pdf(self, mse, r2):
        c = canvas.Canvas(self.pdf_name, pagesize=A4)
        width, height = A4

        # Initial coordinates for image placement
        x, y = 1 * inch, height - 2 * inch  # Start 1 inch from left, 2 inches from top

        # Add visualizations, maintaining original image size and arranging on pages
        image_folders = [self.image_folder, os.path.join(self.image_folder, 'linear_regression_plots')]

        # Add visualizations from all folders, scaling and arranging on pages
        for folder in image_folders:
            for img in os.listdir(folder):
                if img.endswith(".png"):
                    img_path = os.path.join(folder, img)
                    img_obj = Image.open(img_path)
                    img_width, img_height = img_obj.size

                    # Scale down the image to fit two per page
                    scale_factor = (height / 2 - 1 * inch) / img_height
                    new_width = img_width * scale_factor
                    new_height = img_height * scale_factor

                    # Check if there's enough space on the current page
                    if y - new_height < 0.5 * inch:
                        c.showPage()  # Move to the next page
                        x, y = 1 * inch, height - 3 * inch  # Reset coordinates

                    c.drawImage(img_path, x, y - new_height, new_width, new_height)
                    
                    # Move to the right for the next image or new line if needed
                    x += new_width + 0.5 * inch 
                    if x + new_width > width - 1 * inch:  # 1-inch margin at the right
                        x = 1 * inch
                        y -= new_height + 0.5 * inch 

        # Add analysis details (replace with your actual KNN and Linear Regression results)
        c.showPage()
        c.setFont("Helvetica-Bold", 28)
        c.drawString(1 * inch, height - 2 * inch, "KNN Analysis")
        c.setFont("Helvetica", 26)
        c.drawString(1 * inch, height - 4.5 * inch, f"MSE: {mse}") 
        c.drawString(1 * inch, height - 5 * inch, f"R-squared: {r2}")  # Replace with your KNN accuracy
        # ... add more KNN details as needed

        c.setFont("Helvetica-Bold", 28)
        c.drawString(1 * inch, height - 4 * inch, "Linear Regression Analysis")
        c.setFont("Helvetica", 26)
        c.drawString(1 * inch, height - 4.5 * inch, f"MSE: {mse}") 
        c.drawString(1 * inch, height - 5 * inch, f"R-squared: {r2}")
        # ... add more Linear Regression details as needed

        c.save()
        print("Visualization PDF Generated")

    def generate_report(self):
        self.profile_data()
        self.create_visualizations()
        
        #X_train, X_test, y_train, y_test = self.preprocess_data(test_size=0.67, random_state=142)
        LX_train, LX_test, Ly_train, Ly_test = self.preprocess_data(test_size=0.35, random_state=72)
        y_pred, model = self.linear_regression(LX_train, LX_test, Ly_train, Ly_test)
        
        mse, r2 = self.evaluate_model(Ly_test, y_pred)
        # Kmse, Kr2 = self.evaluate_model()
        self.create_report_pdf(mse, r2)
        return self.pdf_name  # Return the visualization PDF filename


# Example usage:
df = pd.read_csv('nissan-dataset1.csv')
print(df.head(15))
report_generator = SurveyGramReport(df, image_folder='images', pdf_name='my_report.pdf')
report_pdf = report_generator.generate_report()
print(f"Report generated: {report_pdf}")