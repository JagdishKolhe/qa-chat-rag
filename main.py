import os
from flask import Flask, render_template
from flask_socketio import SocketIO
import yaml


app = Flask(__name__)
# app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'

data = {}
with open("config.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# print(f"data: {type(data)}, {data}")
app.config.from_object(data)
socketio = SocketIO(app)

os.environ["OPENAI_API_KEY"] = data["OPENAI_API_KEY"]


from qa_openai.qa import qa_openai
qa = None # will do defered init

@app.route('/')
def sessions():
    return render_template('session.html')


@socketio.on('connect_event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print(f"received connect event: {str(json)}")
    global qa   # TODO put qa object in use session for multiuser support, and should not use global variable
    qa = qa_openai()
    qa.setup()

    # socketio.emit('answer_event', {"question": "Connection establised.", "answer" : "", "score": ""})


@socketio.on('ask_event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print(f"received ask event: {str(json)}")
    question = json['question']
    
    result = qa.run_qa_chain(json['model'], question)
    score = round(qa.cal_score(result['query'], result['result']), 2)
    
    socketio.emit('answer_event', {"question": result['query'], "answer" : result['result'], "score": score})
    # socketio.emit('answer_event', {"question": question, "answer" : "Chat backend not available.", "score": 86})
    # socketio.emit('answer_event', json, callback=messageReceived)


if __name__ == '__main__':
    socketio.run(app, debug=True)
