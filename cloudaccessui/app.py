from flask import Flask, render_template

app = Flask(__name__)

# Route to render a simple HTML page
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint returning a simple JSON response
@app.route("/api/details", methods=["GET"])
def api_details():
    return {"message": "This is a simple API endpoint"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
