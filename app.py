from flask import Flask, request,render_template
from flask_cors import CORS, cross_origin
import joblib
import pandas as pd
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = './archivos'  # Actualiza esto con la ruta donde quieres guardar los archivos

# Carga el modelo
modelo = joblib.load('modelo_clasificacion.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Obtiene los datos de la solicitud
    data = request.get_json(force=True)
    print(data)
    # Procesa los datos y utiliza el modelo para hacer una predicción
    obj = data['obj']
    texto = obj['texto']
    texto_procesado = vectorizer.transform([texto])
    prediccion = modelo.predict(texto_procesado)

    # Devuelve la predicción como respuesta
    return {'prediccion': int(prediccion[0])+3}

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html') # Return index.html 

@app.route('/p', methods=['GET'])
def a():
    return 'hola'

@app.route('/predict_file', methods=['POST'])
@cross_origin()
def predict_file():
    # Comprueba si el archivo fue enviado en la solicitud
    if 'file' not in request.files:
        return 'No se encontró ningún archivo en la solicitud'
    
    file = request.files['file']
    
    # Si el usuario no selecciona un archivo, el navegador puede enviar una solicitud vacía sin nombre de archivo
    if file.filename == '':
        return 'No se seleccionó ningún archivo'
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Lee el archivo y procesa los datos
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return 'Tipo de archivo no soportado'
        
        textos = df.iloc[:, 0].tolist()  # Asume que los textos están en la primera columna
        textos_procesados = vectorizer.transform(textos)
        predicciones = modelo.predict(textos_procesados)
        
        # Devuelve las predicciones como respuesta
        return {'predicciones': [{'texto': t, 'clasificacion': int(p)+3} for t, p in zip(textos, predicciones)]}


if __name__ == '__main__':
    app.run(port=5000, debug=True)
    
def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.replace('Ãº','u')
    texto = texto.replace('Ã³','o')
    texto = texto.replace('Ã±','ñ')
    texto = texto.replace('Ã¡','a')
    texto = texto.replace('Ã','i')
    return texto
