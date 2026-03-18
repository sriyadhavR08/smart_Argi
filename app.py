from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import numpy as np
import cv2
import requests

app = Flask(__name__)

classes = ['early_blight','healthy','late_blight']

solutions = {
"early_blight":"Remove infected leaves and use fungicide such as Chlorothalonil.",
"late_blight":"Apply copper based fungicide and avoid overhead watering.",
"healthy":"Your plant is healthy. Maintain proper watering and sunlight."
}

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# 🌦 Weather API
def get_weather():

    api_key = "1c24a15b4c87b0a36189e5d1e72b5958"
    city = "Coimbatore"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    data = requests.get(url).json()

    if "main" in data:
        temperature = data["main"]["temp"]
        weather = data["weather"][0]["description"]
    else:
        temperature = "N/A"
        weather = "Weather data not available"

    return temperature, weather


# 🌱 Fertilizer advice
def fertilizer_advice(temp):

    if temp > 32:
        return "Hot weather. Use potassium rich fertilizer and irrigate regularly."

    elif temp > 25:
        return "Normal weather. Use balanced NPK fertilizer."

    else:
        return "Cool weather. Reduce watering and use organic compost."


# 🌾 Main Page
@app.route('/', methods=['GET','POST'])
def index():

    prediction = None
    solution = None
    confidence = None
    heatmap = None

    temp, weather = get_weather()
    fertilizer = fertilizer_advice(temp)

    if request.method == 'POST':

        file = request.files['file']
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():

            output = model(img)

            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = round(probabilities.max().item()*100,2)

            _, predicted = torch.max(output,1)

        prediction = classes[predicted.item()]
        solution = solutions[prediction]

        img.requires_grad_()
        output = model(img)

        score = output[0][predicted]
        score.backward()

        saliency = img.grad.data.abs().squeeze().cpu().numpy()
        saliency = np.max(saliency, axis=0)

        saliency = (saliency - saliency.min())/(saliency.max()-saliency.min())
        saliency = cv2.resize(saliency,(224,224))

        heatmap = cv2.applyColorMap(np.uint8(255*saliency), cv2.COLORMAP_JET)

        original = cv2.imread(filepath)
        original = cv2.resize(original,(224,224))

        overlay = cv2.addWeighted(original,0.6,heatmap,0.4,0)

        heatmap_path = os.path.join("static","heatmap.jpg")
        cv2.imwrite(heatmap_path,overlay)

    return render_template(
        "index.html",
        prediction=prediction,
        solution=solution,
        confidence=confidence,
        heatmap="heatmap.jpg",
        temperature=temp,
        weather=weather,
        fertilizer=fertilizer
    )


# 🤖 Farmer AI Chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():

    user_message = request.json["message"].lower()

    # Detect Tamil
    tamil_words = ["தக்காளி","அரிசி","உரம்","நோய்","மஞ்சள்"]
    is_tamil = any(word in user_message for word in tamil_words)

    # 🌾 Tomato problem
    if "tomato" in user_message or "தக்காளி" in user_message:

        if "yellow" in user_message or "மஞ்சள்" in user_message:

            if is_tamil:
                reply = "தக்காளி செடியில் மஞ்சள் இலை வந்தால் அது nutrient deficiency அல்லது early blight நோய் ஆக இருக்கலாம். NPK fertilizer பயன்படுத்தவும்."
            else:
                reply = "Yellow leaves in tomato may indicate nutrient deficiency or early blight. Apply balanced NPK fertilizer."

        else:
            if is_tamil:
                reply = "தக்காளி செடிக்கு நல்ல வடிகால் மண், சூரியஒளி மற்றும் முறையான நீர்ப்பாசனம் தேவை."
            else:
                reply = "Tomato plants need well-drained soil, sunlight, and regular watering."

    # 🌾 Rice
    elif "rice" in user_message or "அரிசி" in user_message:

        if is_tamil:
            reply = "அரிசி பயிருக்கு நீர் நிறைந்த வயல் மற்றும் nitrogen fertilizer முக்கியம்."
        else:
            reply = "Rice requires flooded fields and nitrogen fertilizer."

    # 🌱 Fertilizer
    elif "fertilizer" in user_message or "உரம்" in user_message:

        if is_tamil:
            reply = "பயிரின் வகையைப் பொறுத்து NPK அல்லது organic compost பயன்படுத்தலாம்."
        else:
            reply = "Use organic compost or balanced NPK fertilizer depending on the crop."

    # 🍃 Disease
    elif "disease" in user_message or "நோய்" in user_message:

        if is_tamil:
            reply = "மேலே உள்ள leaf image upload செய்து AI மூலம் நோய் கண்டறியலாம்."
        else:
            reply = "Upload the leaf image above to detect the disease using AI."

    else:

        if is_tamil:
            reply = "பயிர், உரம் அல்லது நோய் பற்றிய கேள்விகளை கேளுங்கள்."
        else:
            reply = "Please ask questions about crops, fertilizer, or plant disease."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)