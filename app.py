from flask import Flask, request, render_template
import torch
import torchaudio
from torchvision import models
from utils import preprocess_audio
import os
app = Flask(__name__)

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Genre label mapping
idx_to_genre = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

@app.route("/", methods=["GET", "POST"])
def index():
    genre = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            audio_path = os.path.join("static", file.filename)
            file.save(audio_path)

            input_tensor = preprocess_audio(audio_path)
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0))
                predicted = torch.argmax(output, dim=1).item()
                genre = idx_to_genre[predicted]

    return render_template("index.html", genre=genre)

if __name__ == "__main__":
    app.run(debug=True)
