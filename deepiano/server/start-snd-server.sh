#!/bin/bash


DEEPIANO_DIR=$HOME/deepiano

cd $DEEPIANO_DIR
source .env/bin/activate

cd $DEEPIANO_DIR/deepiano/server

gunicorn -c gunicorn.conf transcribe_server:app
python transcribe_worker.py --model_dir=../../data/models/theone_model_devocal > /data/logs/transcribe_worker.log 2>&1 &
python transcribe_worker.py --model_dir=../../data/models/theone_model_devocal > /data/logs/transcribe_worker.log 2>&1 &
