from flask import Flask, request, session, render_template, redirect, url_for,flash
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "1234" 

# Dictionary to store user information
user_logs = {}

@app.route('/')
def front():
    log_user_info(request)
    return render_template('front.html')

def log_user_info(req):
    user_ip = req.remote_addr
    user_agent = req.user_agent.string
    access_url = req.url
    referrer = req.referrer
    language = req.accept_languages.best

    platform = "Unknown"
    if "Android" in user_agent:
        platform = "Android"
    elif "iPhone" in user_agent or "iPad" in user_agent:
        platform = "iOS"
    elif "Windows" in user_agent:
        platform = "Windows"
    elif "Macintosh" in user_agent:
        platform = "Macintosh"
    elif "Linux" in user_agent:
        platform = "Linux"

    browser = "Unknown"
    if "Edg" in user_agent:
        browser = "Edge"
    elif "Chrome" in user_agent:
        browser = "Chrome"
    elif "Safari" in user_agent:
        browser = "Safari"
    elif "Firefox" in user_agent:
        browser = "Firefox"

    is_mobile = "Mobile" in user_agent
    is_tablet = "Tablet" in user_agent
    is_pc = not (is_mobile or is_tablet)

    print(f"User IP: {user_ip}")
    print(f"User Agent: {user_agent}")
    print(f"Referrer: {referrer}")
    print(f"Language: {language}")
    print(f"Accessed URL: {access_url}")
    print(f"Platform: {platform}")
    print(f"Browser: {browser}")
    print(f"Is Mobile: {is_mobile}")
    print(f"Is Tablet: {is_tablet}")
    print(f"Is PC: {is_pc}")

    with open('user_info.txt', 'a') as f:
        f.write(f"User IP: {user_ip}\n")
        f.write(f"User Agent: {user_agent}\n")
        f.write(f"Referrer: {referrer}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Accessed URL: {access_url}\n")
        f.write(f"Platform: {platform}\n")
        f.write(f"Browser: {browser}\n")
        f.write(f"Is Mobile: {is_mobile}\n")
        f.write(f"Is Tablet: {is_tablet}\n")
        f.write(f"Is PC: {is_pc}\n\n")

@app.route('/userlogs')
def show_user_logs():
    return str(user_logs)

USERS = {
    " ":" ",
    "user1": "password1",
    "user2": "password2"
}

def authenticate(username, password):
    return username in USERS and USERS[username] == password

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if authenticate(username, password):
            session["authenticated"] = Trueimage1_path = os.path.join(os.path.dirname(__file__), "images", "image1.jpg")
            with open('users.txt', 'a') as f:
                f.write(f"User: {username}\n")
                f.write(f"{datetime.now().strftime('%d/%m/%Y_%H:%M:%S')}\n")
            return redirect(url_for("index"))
        else:
            error = "Invalid username or password"
    else:
        error = None
    return render_template("login.html", error=error)


@app.route('/index')
def index():
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route("/analysis")
def analysis():
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    else:
        return render_template("analysis.html")

@app.route('/upload', methods=['GET','POST'])
def upload():
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    else:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename!= '':
                file.save('C:\\Users\\chandan\\Desktop\\22copy\\uploads\\new.csv')
                flash('File uploaded successfully', 'success')
            else:
                flash('No file selected', 'error')
        else:
            flash('No file part', 'error')
        return render_template("upload.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    else:
        if request.method == 'POST':
                    data = CustomData(
                        gender=request.form.get('gender'),
                        attendance=float(request.form.get('attendance')),
                        extra_curricular=request.form.get('extra_curricular'),
                        university_type=request.form.get('university_type'),
                        test_preparation_course=request.form.get('test_preparation_course'),
                        test_score=float(request.form.get('prep_score')),
                        prep_score=float(request.form.get('test_score')),
                        name = request.form.get('name'))
                
                    modelname = request.form.get('name')
                    session["modelname"]=modelname
                    print(modelname)
                    obj = DataIngestion()
                    train_data, test_data = obj.initiate_data_ingestion()

                    data_transformation = DataTransformation()
                    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

                    
                    pred_df = data.get_data_as_data_frame()
                    print("Name:", data.name)

                    print(pred_df.to_string(index=False))
                    print(pred_df.shape)

                    result = PredictPipeline()
                    results=result.predict(modelname,pred_df)

                    res = PredictPipeline()
                    accuracy=res.accuracy(train_arr,test_arr,modelname)
                    print(accuracy*100,'%')

                    if results[0] > 40:
                        result = "PASS"
                    else:
                        result = "FAIL"

                    return render_template('prediction_results.html', results=results,accuracy=accuracy*100,result=result)
            
        else:
            return render_template('home.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('front'))

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)