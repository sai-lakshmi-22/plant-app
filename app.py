from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    session,
    flash
)

from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from keras.models import load_model

from PIL import Image

import numpy as np
import json
import os
import requests

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_key")

# DATABASE
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# USER TABLE
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True)
    mobile = db.Column(db.String(20))
    password = db.Column(db.String(250))

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120))
    prediction = db.Column(db.String(120))
    confidence = db.Column(db.String(50))
    created = db.Column(db.DateTime, default=db.func.now())

# MODEL
MODEL_PATH = "model/fixed_plant_model.h5"
LABEL_PATH = "model/labels.json"

model = load_model(MODEL_PATH, compile=False)

with open(LABEL_PATH) as f:
    class_indices = json.load(f)

labels = {v:k for k,v in class_indices.items()}

print("\n===== MODEL CLASSES =====")

for i in range(len(labels)):
    print(i, "=>", labels[i])

print("=========================\n")

treatments = {

    "Apple___Apple_scab": {
    "English":"Spray fungicide. Remove infected leaves. Improve airflow around tree.",
    "हिंदी":"फफूंदनाशक छिड़कें। संक्रमित पत्ते हटाएँ। पेड़ के आसपास हवा बढ़ाएँ।",
    "தமிழ்":"பூஞ்சை மருந்து தெளிக்கவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும். காற்றோட்டம் அதிகரிக்கவும்.",
    "తెలుగు":"ఫంగిసైడ్ పిచికారీ చేయండి. దెబ్బతిన్న ఆకులు తొలగించండి. గాలి ప్రవాహం పెంచండి."
    },

    "Apple___Black_rot": {
    "English":"Prune infected branches. Apply copper fungicide. Remove fallen fruit.",
    "हिंदी":"संक्रमित शाखाएँ काटें। कॉपर फफूंदनाशक लगाएँ। गिरे फल हटाएँ।",
    "தமிழ்":"பாதிக்கப்பட்ட கிளைகளை வெட்டவும். காப்பர் மருந்து பயன்படுத்தவும்.",
    "తెలుగు":"సంఖ్యమిత కొమ్మలు తొలగించండి. కాపర్ ఫంగిసైడ్ వాడండి."
    },

    "Apple___Cedar_apple_rust": {
    "English":"Spray fungicide in spring. Remove nearby cedar hosts if possible.",
    "हिंदी":"वसंत में फफूंदनाशक छिड़कें। पास के सीडर पौधे हटाएँ।",
    "தமிழ்":"வசந்தத்தில் மருந்து தெளிக்கவும். அருகிலுள்ள சீடர் செடிகளை அகற்றவும்.",
    "తెలుగు":"వసంతంలో ఫంగిసైడ్ వాడండి. దగ్గరలోని సీడర్ మొక్కలు తొలగించండి."
    },

    "Apple___healthy": {
    "English":"Plant is healthy. Continue balanced nutrition and watering.",
    "हिंदी":"पौधा स्वस्थ है। नियमित पोषण और पानी दें।",
    "தமிழ்":"தாவரம் ஆரோக்கியமாக உள்ளது. சீரான நீர் மற்றும் ஊட்டச்சத்து வழங்கவும்.",
    "తెలుగు":"మొక్క ఆరోగ్యంగా ఉంది. సరైన నీరు మరియు పోషణ ఇవ్వండి."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
    "English":"Use resistant seeds. Rotate crops. Apply fungicide if severe.",
    "हिंदी":"प्रतिरोधी बीज उपयोग करें। फसल चक्र अपनाएँ। जरूरत हो तो दवा छिड़कें।",
    "தமிழ்":"எதிர்ப்பு விதைகள் பயன்படுத்தவும். பயிர் மாற்றம் செய்யவும்.",
    "తెలుగు":"రోగనిరోధక విత్తనాలు వాడండి. పంట మార్పిడి చేయండి."
    },

    "Corn_(maize)___Common_rust_": {
    "English":"Use rust-resistant varieties. Apply fungicide when needed.",
    "हिंदी":"रस्ट-प्रतिरोधी किस्में लगाएँ। जरूरत पर दवा छिड़कें।",
    "தமிழ்":"ரஸ்ட் எதிர்ப்பு வகைகள் பயிரிடவும்.",
    "తెలుగు":"రస్ట్ నిరోధక రకాలు వాడండి."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
    "English":"Use crop rotation. Destroy residue. Apply foliar fungicide.",
    "हिंदी":"फसल चक्र अपनाएँ। अवशेष नष्ट करें। पत्तों पर दवा छिड़कें।",
    "தமிழ்":"பயிர் மாற்றம் செய்யவும். கழிவுகளை அகற்றவும்.",
    "తెలుగు":"పంట మార్పిడి చేయండి. అవశేషాలు తొలగించండి."
    },

    "Corn_(maize)___healthy": {
    "English":"Corn crop is healthy. Maintain fertilizer and irrigation schedule.",
    "हिंदी":"मक्का स्वस्थ है। खाद और सिंचाई नियमित रखें।",
    "தமிழ்":"மக்காச்சோளம் ஆரோக்கியமாக உள்ளது.",
    "తెలుగు":"మొక్కజొన్న ఆరోగ్యంగా ఉంది."
    },

    "Pepper__bell___Bacterial_spot": {
    "English":"Use copper spray. Avoid overhead watering. Remove infected leaves.",
    "हिंदी":"कॉपर स्प्रे करें। ऊपर से पानी न दें। संक्रमित पत्ते हटाएँ।",
    "தமிழ்":"காப்பர் தெளிக்கவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    "తెలుగు":"కాపర్ స్ప్రే చేయండి. దెబ్బతిన్న ఆకులు తొలగించండి."
    },

    "Pepper__bell___healthy": {
    "English":"Pepper plant is healthy. Continue proper watering and nutrition.",
    "हिंदी":"मिर्च पौधा स्वस्थ है। पानी और पोषण जारी रखें।",
    "தமிழ்":"மிளகாய் செடி ஆரோக்கியமாக உள்ளது.",
    "తెలుగు":"మిరప మొక్క ఆరోగ్యంగా ఉంది."
    },

    "Potato___Early_blight": {
    "English":"Spray fungicide. Remove infected lower leaves. Rotate crops yearly.",
    "हिंदी":"दवा छिड़कें। नीचे की संक्रमित पत्तियाँ हटाएँ। फसल बदलें।",
    "தமிழ்":"மருந்து தெளிக்கவும். கீழ் இலைகளை அகற்றவும்.",
    "తెలుగు":"ఫంగిసైడ్ వాడండి. కింద ఆకులు తొలగించండి."
    },

    "Potato___Late_blight": {
    "English":"Apply fungicide immediately. Reduce humidity. Remove infected plants.",
    "हिंदी":"तुरंत दवा लगाएँ। नमी कम करें। संक्रमित पौधे हटाएँ।",
    "தமிழ்":"உடனே மருந்து பயன்படுத்தவும். பாதிக்கப்பட்ட செடிகளை அகற்றவும்.",
    "తెలుగు":"తక్షణం ఫంగిసైడ్ వాడండి. దెబ్బతిన్న మొక్కలు తొలగించండి."
    },

    "Potato___healthy": {
    "English":"Potato plant is healthy. Continue proper care.",
    "हिंदी":"आलू पौधा स्वस्थ है। देखभाल जारी रखें।",
    "தமிழ்":"உருளைக்கிழங்கு செடி ஆரோக்கியமாக உள்ளது.",
    "తెలుగు":"బంగాళాదుంప మొక్క ఆరోగ్యంగా ఉంది."
    },

    "Tomato_Bacterial_spot": {
    "English":"Use copper bactericide. Remove damaged leaves. Keep foliage dry.",
    "हिंदी":"कॉपर दवा लगाएँ। खराब पत्ते हटाएँ। पत्ते सूखे रखें।",
    "தமிழ்":"காப்பர் மருந்து பயன்படுத்தவும். இலைகளை உலர வைக்கவும்.",
    "తెలుగు":"కాపర్ మందు వాడండి. ఆకులు పొడిగా ఉంచండి."
    },

    "Tomato_Early_blight": {
    "English":"Apply fungicide. Mulch soil. Remove infected leaves.",
    "हिंदी":"दवा लगाएँ। मल्च करें। संक्रमित पत्ते हटाएँ।",
    "தமிழ்":"மருந்து தெளிக்கவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    "తెలుగు":"ఫంగిసైడ్ వాడండి. దెబ్బతిన్న ఆకులు తొలగించండి."
    },

    "Tomato_Late_blight": {
    "English":"Use fungicide urgently. Improve spacing. Remove infected plants.",
    "हिंदी":"तुरंत दवा दें। दूरी रखें। संक्रमित पौधे हटाएँ।",
    "தமிழ்":"உடனே மருந்து பயன்படுத்தவும். பாதிக்கப்பட்ட செடிகளை அகற்றவும்.",
    "తెలుగు":"తక్షణం మందు వాడండి. దెబ్బతిన్న మొక్కలు తొలగించండి."
    },

    "Tomato_Leaf_Mold": {
    "English":"Increase ventilation. Lower humidity. Apply suitable fungicide.",
    "हिंदी":"हवा बढ़ाएँ। नमी कम करें। उचित दवा लगाएँ।",
    "தமிழ்":"காற்றோட்டம் அதிகரிக்கவும். ஈரப்பதம் குறைக்கவும்.",
    "తెలుగు":"గాలి పెంచండి. తేమ తగ్గించండి."
    },

    "Tomato_Septoria_leaf_spot": {
    "English":"Remove infected leaves. Mulch soil. Use fungicide spray.",
    "हिंदी":"संक्रमित पत्ते हटाएँ। मल्च करें। दवा छिड़कें।",
    "தமிழ்":"பாதிக்கப்பட்ட இலைகளை அகற்றவும். மருந்து தெளிக்கவும்.",
    "తెలుగు":"దెబ్బతిన్న ఆకులు తొలగించండి. మందు పిచికారీ చేయండి."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
    "English":"Spray neem oil or miticide. Wash underside of leaves.",
    "हिंदी":"नीम तेल या दवा छिड़कें। पत्तों के नीचे साफ करें।",
    "தமிழ்":"வேப்பெண்ணெய் தெளிக்கவும். இலைகள் கீழ்பகுதியை சுத்தம் செய்யவும்.",
    "తెలుగు":"వేపనూనె స్ప్రే చేయండి. ఆకుల కింద భాగం శుభ్రం చేయండి."
    },

    "Tomato__Target_Spot": {
    "English":"Apply fungicide. Improve airflow. Remove infected leaves.",
    "हिंदी":"दवा लगाएँ। हवा बढ़ाएँ। संक्रमित पत्ते हटाएँ।",
    "தமிழ்":"மருந்து தெளிக்கவும். காற்றோட்டம் அதிகரிக்கவும்.",
    "తెలుగు":"ఫంగిసైడ్ వాడండి. గాలి పెంచండి."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
    "English":"Control whiteflies. Remove infected plants. Use resistant seeds.",
    "हिंदी":"सफेद मक्खी नियंत्रित करें। संक्रमित पौधे हटाएँ।",
    "தமிழ்":"வைட் ஃப்ளை கட்டுப்படுத்தவும். பாதிக்கப்பட்ட செடிகளை அகற்றவும்.",
    "తెలుగు":"వైట్ ఫ్లై నియంత్రించండి. దెబ్బతిన్న మొక్కలు తొలగించండి."
    },

    "Tomato__Tomato_mosaic_virus": {
    "English":"Remove infected plants. Disinfect tools. Avoid contamination.",
    "हिंदी":"संक्रमित पौधे हटाएँ। औजार साफ करें।",
    "தமிழ்":"பாதிக்கப்பட்ட செடிகளை அகற்றவும். கருவிகளை சுத்தம் செய்யவும்.",
    "తెలుగు":"దెబ్బతిన్న మొక్కలు తొలగించండి. పరికరాలు శుభ్రం చేయండి."
    },

    "Tomato_healthy": {
    "English":"Tomato plant is healthy. Maintain regular fertilizer and watering.",
    "हिंदी":"टमाटर पौधा स्वस्थ है। खाद और पानी जारी रखें।",
    "தமிழ்":"தக்காளி செடி ஆரோக்கியமாக உள்ளது.",
    "తెలుగు":"టమాటా మొక్క ఆరోగ్యంగా ఉంది."
    }

}

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# CREATE DB
with app.app_context():
    db.create_all()
# =====================================
# Put this ABOVE @app.route("/")
# in app.py
# Full clean preprocess function
# =====================================

def preprocess(img):

    img = img.resize(
        (224, 224)
    )

    img = np.array(
        img,
        dtype=np.float32
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img
# ROUTES
@app.route("/")
def splash():
    return render_template("splash.html")

@app.route("/intro")
def intro():
    return render_template("intro.html")

@app.route("/language")
def language():
    return render_template("language.html")

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user"] = user.name
            return redirect("/home")

        flash("Invalid Login")
        return redirect("/login")

    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        mobile = request.form["mobile"]
        password = request.form["password"]

        existing = User.query.filter_by(email=email).first()

        if existing:
            flash("Email already exists")
            return redirect("/register")

        hashed = generate_password_hash(password)

        user = User(
            name=name,
            email=email,
            mobile=mobile,
            password=hashed
        )

        db.session.add(user)
        db.session.commit()

        flash("Account Created")
        return redirect("/login")

    return render_template("register.html")

@app.route("/home")
def home():

    if "user" not in session:
        return redirect("/login")

    return render_template(
        "home.html",
        username=session["user"]
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        })

    file = request.files["image"]

    try:
        # read image
        img = Image.open(file).convert("RGB")
        img = preprocess(img)

        # predict
        pred = model.predict(img)

        index = int(np.argmax(pred))
        confidence = float(np.max(pred))

        result = labels[index]
        score = round(confidence * 100, 2)

        # selected language
        lang = request.form.get(
            "lang",
            "English"
        )

        # treatment lookup
        item = treatments.get(
            result,
            {}
        )

        advice = item.get(
            lang,
            item.get(
                "English",
                "Consult agriculture expert."
            )
        )

        # clean disease name
        nice_name = result.replace(
            "___",
            " - "
        ).replace(
            "__",
            " "
        ).replace(
            "_",
            " "
        )

        # save history if logged in
        if "user" in session:

            history = History(
                username=session["user"],
                prediction=nice_name,
                confidence=str(score) + "%"
            )

            db.session.add(history)
            db.session.commit()

        return jsonify({
            "success": True,
            "prediction": nice_name,
            "confidence": score,
            "treatment": advice
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })
    
@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    items = History.query.filter_by(
        username=session["user"]
    ).order_by(
        History.id.desc()
    ).all()

    return render_template(
        "history.html",
        items=items,
        username=session["user"]
    )

@app.route("/treatment")
def treatment():

    disease_list = []

    for key in treatments.keys():

        nice = key.replace(
            "___", " - "
        ).replace(
            "__", " "
        ).replace(
            "_", " "
        )

        disease_list.append({
            "raw": key,
            "nice": nice
        })

    disease_list.sort(
        key=lambda x: x["nice"]
    )

    return render_template(
        "treatment.html",
        diseases=disease_list
    )

@app.route("/get-treatment", methods=["POST"])
def get_treatment():

    data = request.get_json()

    disease = data.get("disease", "")
    lang = data.get("lang", "English")

    # Treatment text
    item = treatments.get(disease, {})

    advice = item.get(
        lang,
        item.get(
            "English",
            "Consult agriculture expert."
        )
    )

    # Clean display name
    nice_name = disease.replace(
        "___", " - "
    ).replace(
        "__", " "
    ).replace(
        "_", " "
    )

    # Default multilingual medicine
    medicine_map = {
        "English":"Copper fungicide / Neem spray",
        "हिंदी":"कॉपर फफूंदनाशक / नीम स्प्रे",
        "தமிழ்":"காப்பர் பூஞ்சை மருந்து / வேப்பெண்ணெய் தெளிப்பு",
        "తెలుగు":"కాపర్ ఫంగిసైడ్ / వేప స్ప్రే"
    }

    # Default multilingual prevention
    prevention_map = {
        "English":"Keep field clean and inspect leaves weekly.",
        "हिंदी":"खेत साफ रखें और पत्तों की साप्ताहिक जाँच करें।",
        "தமிழ்":"பயிர் நிலத்தை சுத்தமாக வைத்து வாரம் ஒருமுறை இலைகளை பார்க்கவும்.",
        "తెలుగు":"పొలం శుభ్రంగా ఉంచి వారానికి ఒకసారి ఆకులను పరిశీలించండి."
    }

    severity = "Moderate"

    # Healthy case
    if "healthy" in disease.lower():

        medicine_map = {
            "English":"No medicine needed",
            "हिंदी":"दवा की आवश्यकता नहीं",
            "தமிழ்":"மருந்து தேவையில்லை",
            "తెలుగు":"మందు అవసరం లేదు"
        }

        prevention_map = {
            "English":"Continue watering and nutrition.",
            "हिंदी":"पानी और पोषण जारी रखें।",
            "தமிழ்":"நீர் மற்றும் ஊட்டச்சத்து தொடரவும்.",
            "తెలుగు":"నీరు మరియు పోషణ కొనసాగించండి."
        }

        severity = "None"

    elif "virus" in disease.lower():

        medicine_map = {
            "English":"No direct cure. Remove infected plants.",
            "हिंदी":"सीधा इलाज नहीं। संक्रमित पौधे हटाएँ।",
            "தமிழ்":"நேரடி சிகிச்சை இல்லை. பாதிக்கப்பட்ட செடிகளை அகற்றவும்.",
            "తెలుగు":"నేరుగా మందు లేదు. దెబ్బతిన్న మొక్కలు తొలగించండి."
        }

        prevention_map = {
            "English":"Control insects and disinfect tools.",
            "हिंदी":"कीट नियंत्रित करें और औजार साफ रखें।",
            "தமிழ்":"பூச்சிகளை கட்டுப்படுத்து கருவிகளை சுத்தப்படுத்தவும்.",
            "తెలుగు":"పురుగులు నియంత్రించి పరికరాలు శుభ్రం చేయండి."
        }

        severity = "High"

    elif "blight" in disease.lower():

        medicine_map = {
            "English":"Mancozeb / Copper fungicide",
            "हिंदी":"मैनकोजेब / कॉपर फफूंदनाशक",
            "தமிழ்":"மாங்கோசெப் / காப்பர் மருந்து",
            "తెలుగు":"మాంకోజెబ్ / కాపర్ ఫంగిసైడ్"
        }

        prevention_map = {
            "English":"Reduce moisture and remove infected leaves.",
            "हिंदी":"नमी कम करें और संक्रमित पत्ते हटाएँ।",
            "தமிழ்":"ஈரப்பதம் குறைத்து பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
            "తెలుగు":"తేమ తగ్గించి దెబ్బతిన్న ఆకులు తొలగించండి."
        }

        severity = "High"

    medicine = medicine_map.get(
        lang,
        medicine_map["English"]
    )

    prevention = prevention_map.get(
        lang,
        prevention_map["English"]
    )

    return jsonify({
        "success": True,
        "name": nice_name,
        "treatment": advice,
        "medicine": medicine,
        "prevention": prevention,
        "severity": severity
    })
@app.route("/reports")
def reports():

    if "user" not in session:
        return redirect("/login")

    username = session["user"]

    items = History.query.filter_by(
        username=username
    ).all()

    total = len(items)

    healthy = 0
    diseased = 0
    confidence_sum = 0

    latest = "No scans yet"

    if total > 0:

        latest = items[-1].prediction

    for item in items:

        conf = float(
            item.confidence.replace("%","")
        )

        confidence_sum += conf

        if "healthy" in item.prediction.lower():
            healthy += 1
        else:
            diseased += 1

    avg = round(
        confidence_sum / total,
        2
    ) if total > 0 else 0

    return render_template(
        "reports.html",
        username=username,
        total=total,
        healthy=healthy,
        diseased=diseased,
        latest=latest,
        avg=avg,
        items=items[-5:]
    )

@app.route("/weather")
def weather():

    if "user" not in session:
        return redirect("/login")

    return render_template(
        "weather.html",
        username=session["user"]
    )

@app.route("/weather")
def weather():

    if "user" not in session:
        return redirect("/login")

    # dynamic city from URL
    city = request.args.get(
        "city",
        "Visakhapatnam"
    )

    api_key = "268780d436f67798759c93ea770e94db"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:

        res = requests.get(url).json()

        if res.get("cod") != 200:

            temp = "--"
            humidity = "--"
            wind = "--"
            condition = "Unavailable"

        else:

            temp = round(
                res["main"]["temp"]
            )

            humidity = res["main"]["humidity"]

            wind = res["wind"]["speed"]

            condition = res["weather"][0]["main"]

    except:

        temp = "--"
        humidity = "--"
        wind = "--"
        condition = "Offline"

    return render_template(
        "weather.html",
        username=session["user"],
        city=city,
        temp=temp,
        humidity=humidity,
        wind=wind,
        condition=condition
    )

if __name__ == "__main__":
    app.run(debug=True)