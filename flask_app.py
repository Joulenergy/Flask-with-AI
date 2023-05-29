from flask import Flask, request
from pathlib import Path
import tensorflow as tf

from clean_text import clean_texts

CWD = Path(file).parent.resolve()
model = tf.keras.models.load_model(CWD/'models/cyberbullying-bdlstm.h5')
with open(CWD/'models/tokenizer.json') as file:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(file.read())

app = Flask(name)

@app.route('/', methods=['GET'])
def main():
    return 'Hello world'

@app.route('/echo', methods=['GET', 'POST'])
def echo():
    if 'echo' in request.args:
        return request.args.get('echo')
    else:
        return "Param not found", 404

@app.route('/cyberbullying', methods=['GET'])
def cyberbullying():
    messages = clean_texts([request.args.get('msg')], tokenizer)
    return request.args.get('msg')+'<br/>'+str(model.predict(messages).tolist()[0][0])

if __name__ == '__main__':
    app.run()
