import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
os.environ['CHORD_DESKTOP_DATA']='D:/ChordStudio/data'
from app.web.server import create_app
from fastapi import HTTPException
import uvicorn
class PreviewJobs:
    def __init__(self,store):self.store=store
    def submit(self,*a,**k):raise HTTPException(403,'This preview cannot submit model jobs')
    def shutdown(self):pass
uvicorn.run(create_app('D:/ChordStudio/data/projects',manager_factory=PreviewJobs),host='127.0.0.1',port=61858,log_level='warning')
