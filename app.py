from flask import Flask,render_template, request
import pickle

import re
import numpy as np
import pandas as pd
import spacy
# import spacy_lookup
import json 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

app = Flask(__name__)
nlp = spacy.blank('id')
customize_stop_words=['terima','kasih','kepada','yth','terima kasih','dan','nya','mohon','tindak','lanjut','tindaklanjuti']
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        judul = request.form['judul']
        judul_cleaned = preproses(judul)
        print(judul_cleaned)
        judul_seq = tokenize_judul(judul_cleaned)

        isi = request.form['isi']
        isi_cleaned = preproses(isi)
        print(isi_cleaned)
        isi_seq = tokenize_isi(isi_cleaned)

        id_instansi = predict(judul_seq,isi_seq)[0]
        nama_instansi = encode(id_instansi)
        return render_template("hasil.html",judul=judul,isi=isi,id_instansi=id_instansi,nama_instansi=nama_instansi)

    else:
        return render_template("index.html")

def preproses(text):
    with open('indo_slang_word.txt') as f: 
        data = f.read() 
    indo_slang = json.loads(data)
    text = text.lower()
    text = ' '.join(list(map(indo_slang.get, text.split(), text.split())))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I|re.A) # if expression in the sentence is not a word then this code change them to space
    text = text.strip() # remove extra whitespace
    text = re.sub(' +', ' ', text)
    text = nlp(text)
    print('sebelum:',text)
    text = [token.lemma_ for token in text if not token.is_stop] 
    print('sesudah:',text)
    print('\n')
    return text

def tokenize_judul(judul_cleaned):
    with open('tokenizer_judul.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    judul_seq = tokenizer.texts_to_sequences([judul_cleaned])
    judul_seq = pad_sequences(judul_seq, padding='post',maxlen=10)
    return judul_seq

def tokenize_isi(isi_cleaned):
    with open('tokenizer_isi.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    isi_seq = tokenizer.texts_to_sequences([isi_cleaned])
    isi_seq = pad_sequences(isi_seq, padding='post',maxlen=20)
    return isi_seq

def predict(judul,isi):    
    model = keras.models.load_model('combined_model.h5')
    target_pred = model.predict([judul,isi])
    instance_id = np.argmax(target_pred,axis=1) + 1
    return instance_id

def encode(id):
    encoding={
        1:"Badan Penyelenggara Jaminan Sosial Kesehatan",
        2:"Pemerintah Provinsi DKI Jakarta",
        3:"Kementerian Sosial",
        4:"Pemerintah Kota Bandung",
        5:"Pemerintah Kota Semarang",
        6:"Kementerian Pendidikan dan Kebudayaan",
        7:"Kementerian Badan Usaha Milik Negara",
        8:"Kementerian Hukum dan HAM",
        9:"Pemerintah Kabupaten Bojonegoro",
        10:"Kementerian Dalam Negeri",
        11:"Badan Kepegawaian Negara",
        12:"Kepolisian Republik Indonesia"
        } 
    return encoding[id]

if __name__ == "__main__":
    app.run(debug=True)