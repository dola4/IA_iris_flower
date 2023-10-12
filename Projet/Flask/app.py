from Flask import flask, requests;
from distutils import debug

app = flask(__name__) # create FLASK object

#define routes
app.route('/model',methods=['POST'])
def hello_World():
    request_data = requests.get_json(force = True)
    model_name = request_data['model']
    return "You are requesting for a {0} model", format(model_name)


if __name__ == '__main__':
    app.run(port = 8006, debug = True)