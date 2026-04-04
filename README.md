AI Personalized Learning Path Generator

An intelligent system that uses Machine Learning to analyze student performance and generate personalized learning recommendations to improve learning outcomes and engagement.

Features-

Student Performance Analysis
Machine Learning-based Predictions (Random Forest)
Classification of Students (Weak / Intermediate / Strong)
Personalized Learning Recommendations
Interactive Dashboard (Streamlit / Flask)

How It Works-
1)Student Data Collection
  Marks, Quiz Scores, Assignments
  Attendance, Study Time
  Learning Style
2)Data Preprocessing
  Missing Value Handling
  Encoding (Categorical to Numeric)
  Normalization
3)Model Training
  Random Forest Algorithm
4)Performance Analysis
  Weak
  Intermediate
  Strong
5)Recommendation Engine
  Rule-based + ML Insights
6)Generate Personalized Learning Path

Tech Stack-
Frontend: Streamlit / Flask
Backend: Python
Libraries:
  Pandas
  NumPy
  Scikit-learn
ML Algorithm: Random Forest

Machine Learning Details-
Input Features:
  Age
  Gender
  Education Level
  Learning Style
  Previous GPA
  Completed Modules
  Engagement Score
Model Used:
  Random Forest Classifier
Output:
  Student Level (Weak / Intermediate / Strong)
  Personalized Recommendations
  
Application Screenshots-
Login Page-
  <img width="1470" height="783" alt="Screenshot 2026-04-05 at 12 03 13 AM" src="https://github.com/user-attachments/assets/9e3d6bb3-a053-4b5c-aed8-86b277d617d6" />
Dashboard-
  <img width="1469" height="835" alt="Screenshot 2026-04-04 at 6 59 43 PM" src="https://github.com/user-attachments/assets/cf5d3436-de3a-458b-a86a-809d8c8f3241" />
Analysis-
  <img width="1358" height="814" alt="Screenshot 2026-04-04 at 7 02 47 PM" src="https://github.com/user-attachments/assets/92f12e48-6794-4793-806e-24aaae3bfaa7" />
Topic Analysis-
  <img width="587" height="573" alt="Screenshot 2026-04-05 at 12 36 47 AM" src="https://github.com/user-attachments/assets/72897a15-d3a6-4642-90d1-58ad4dc965ed" />
Recommendation Output
  <img width="362" height="732" alt="Screenshot 2026-04-05 at 12 40 10 AM" src="https://github.com/user-attachments/assets/497c14fd-9706-4c7b-a9b8-f8581dbd1ad3" />

Project Structure
├── app.py                 # Main application
├── train_model.py         # Model training script
├── data/                  # Dataset files
├── screenshots/           # App images
├── dashboard.py           #UI
