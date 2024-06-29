from flask import Flask

app = Flask (__name__)

@app.route ('/Home')

def hello_world():

    return 'Hello, World!'

