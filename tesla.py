import pandas as pd

df = pd.read_csv("tesla.csv")

print(df.head())
df.info()

#Cleaning the data

df.isnull().sum()

df= df.dropna()


df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace = True)

print(df.head())

#Visualize the Data

import plotly.express as px

# Create an interactive line chart for 'Close' stock price
fig = px.line(df, x=df.index, y='Close', title='Tesla Stock Price Over Time')
fig.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Stock Features')
plt.show()


# Save the heatmap image
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Stock Features')
plt.savefig('heatmap.png')


import cv2
import numpy as np

# Load the saved heatmap image
img = cv2.imread('heatmap.png')

# Convert the image to grayscale for processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection using the Canny edge detector
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Display the edges detected in the image
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the result
cv2.imwrite('edges_heatmap.png', edges)
