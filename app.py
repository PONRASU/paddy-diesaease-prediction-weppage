from flask import Flask, render_template, request, send_file
from roboflow import Roboflow
# roboflow api 
rf = Roboflow(api_key="PO4GNPqNvVVEr9mDclHV")
project = rf.workspace().project("rise")
model = project.version(1).model

app = Flask("rice")


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/api", methods=["POST"])
def api():
    file = request.files["img"]
    file.save('download.jpg')
    model.predict("./download.jpg", confidence=50, overlap=30).save()
    return send_file("./predictions.jpg")


if __name__ == "__main__":
    app.run(debug=True)