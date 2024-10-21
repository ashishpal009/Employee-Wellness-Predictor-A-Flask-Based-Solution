from flask import Flask, request, render_template
import pickle  # or any other method you are using to load your model

app = Flask(__name__)

# Load your model (make sure to adjust the path)
model = pickle.load(open('lr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Make sure this matches your HTML file name

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    self_employed = int(request.form['self_employed'])
    family_history = int(request.form['family_history'])
    work_interfere = int(request.form['work_interfere'])
    no_employees = int(request.form['no_employees'])
    remote_work = int(request.form['remote_work'])
    tech_company = int(request.form['tech_company'])
    benefits = int(request.form['benefits'])
    care_options = int(request.form['care_options'])
    wellness_program = int(request.form['wellness_program'])
    seek_help = int(request.form['seek_help'])
    anonymity = int(request.form['anonymity'])
    leave = int(request.form['leave'])
    mental_health_consequence = int(request.form['mental_health_consequence'])
    phys_health_consequence = int(request.form['phys_health_consequence'])
    coworkers = int(request.form['coworkers'])
    supervisor = int(request.form['supervisor'])
    mental_health_interview = int(request.form['mental_health_interview'])
    phys_health_interview = int(request.form['phys_health_interview'])
    mental_vs_physical = int(request.form['mental_vs_physical'])
    obs_consequence = int(request.form['obs_consequence'])

    # Create a feature array for prediction
    features = [[age, gender, self_employed, family_history, work_interfere,
                 no_employees, remote_work, tech_company, benefits,
                 care_options, wellness_program, seek_help, anonymity,
                 leave, mental_health_consequence, phys_health_consequence,
                 coworkers, supervisor, mental_health_interview,
                 phys_health_interview, mental_vs_physical, obs_consequence]]

    # Make prediction
    prediction = model.predict(features)[0]

    # Map prediction to 'Yes' or 'No'
    prediction_text = "Yes" if prediction == 1 else "No"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
