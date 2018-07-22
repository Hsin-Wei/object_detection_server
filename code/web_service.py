#!todo-api/flask/bin/python
from flask import Flask, request, redirect, url_for
from  werkzeug  import  secure_filename

import os
import json
import datetime
import shutil

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


import sys
path = ['', '/usr/local', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/root/.local/lib/python2.7/site-packages', '/detectron/lib', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/local/lib/python2.7/dist-packages/IPython/extensions', '/root/.ipython']

for p in path:
    sys.path.append(p)

#Parameters
UPLOAD_FOLDER = '/root/object_detection_server/code/datas/research/imgs'
RESULT_FOLDER = '/root/object_detection_server/code/datas/research/mask_imgs'
ALLOWED_EXTENSIONS  =  set (['jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
app.config['RESULT_FOLDER'] =  RESULT_FOLDER

def  allowed_file ( filename ): 
    return  '.'  in  filename  and filename.rsplit ( '.' ,  1 )[ 1 ]  in  ALLOWED_EXTENSIONS

@app.route ( '/test_web' ,  methods = [ 'GET' ,  'POST' ]) 
def test ():
    return 'Hellow'

@app.route ( '/deletefiles' ,  methods = [ 'GET' ,  'POST' ]) 
def delete_file ():
    try:
        now = datetime.datetime.now()
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
        return 'remove success'
    except:
        return 'error in server side'

@app.route ( '/update' ,  methods = [ 'GET' ,  'POST' ]) 
def upload_file ():
    try:
        now = datetime.datetime.now()
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        if  request.method  ==  'POST' : 
            file  =  request.files['file'] 
            if  file  and  allowed_file ( file.filename ): 
                filename  =  secure_filename ( file.filename ) 
                file.save(os.path.join(app.config[ 'UPLOAD_FOLDER' ],  filename))
        return 'success'
    except:
        return 'error in server side'
    

# @app.route('/result/coordinate', methods=['GET'])
# def result_coordinate():
#     now = datetime.datetime.now()
#     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
#     app.config['RESULT_FOLDER'] = RESULT_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
    
#     if not os.path.exists(app.config['RESULT_FOLDER']):
#             os.makedirs(app.config['RESULT_FOLDER'])
    
#     os.system('/root/object_detection_server/code/infer_for_web.py \
#     --cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
#     --output-dir ' + app.config['RESULT_FOLDER'] + '\
#     --image-ext jpg \
#     --wts /root/object_detection_server/code/model/model_final.pkl \
#     ' + app.config['UPLOAD_FOLDER'])
    
#     with open (app.config['RESULT_FOLDER'] + '/output.json', "r") as myfile:
#         data=myfile.read()
    
#     return data


@app.route('/result/track_bottle', methods=['GET'])
def result_bottle_track():
    now = datetime.datetime.now()
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
    app.config['RESULT_FOLDER'] = RESULT_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
    
    if not os.path.exists(app.config['RESULT_FOLDER']):
            os.makedirs(app.config['RESULT_FOLDER'])
    
    os.system('/root/object_detection_server/code/infer_track_bottle_web.py \
    --cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir ' + app.config['RESULT_FOLDER'] + '\
    --image-ext jpg \
    --wts /root/object_detection_server/code/model/model_final.pkl \
    ' + app.config['UPLOAD_FOLDER'])
    
    with open (app.config['RESULT_FOLDER'] + '/output.json', "r") as myfile:
        data=myfile.read()
    
    return data

@app.route('/result/track_all', methods=['GET'])
def result_all_track():
    now = datetime.datetime.now()
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
    app.config['RESULT_FOLDER'] = RESULT_FOLDER + '/' + ('%s%s%s' % (now.year, now.month, now.day))
    
    if not os.path.exists(app.config['RESULT_FOLDER']):
            os.makedirs(app.config['RESULT_FOLDER'])
    
    os.system('/root/object_detection_server/code/infer_track_all_web.py \
    --cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir ' + app.config['RESULT_FOLDER'] + '\
    --image-ext jpg \
    --interested-object "40,74" \
    --wts /root/object_detection_server/code/model/model_final.pkl \
    ' + app.config['UPLOAD_FOLDER'])
    
    with open (app.config['RESULT_FOLDER'] + '/output.json', "r") as myfile:
        data=myfile.read()
    
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8889, debug=True)