from flask import Flask, request, jsonify
from preprocessor import MedicalTextPreprocessor

app = Flask(__name__)
preprocessor = MedicalTextPreprocessor()

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide a "text" field in JSON body'}), 400
        
    raw_text = data['text']
    cleaned_text = preprocessor.clean_text(raw_text)
    
    return jsonify({
        'status': 'success',
        'raw_text': raw_text,
        'cleaned_text': cleaned_text
    })

if __name__ == '__main__':
    # تشغيل الـ API على Port مختلف لكي لا يتصادم مع باقي الـ APIs (مثلاً Port 5001)
    app.run(host='0.0.0.0', port=5001, debug=True)