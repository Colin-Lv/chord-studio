"""Piano MIDI API: jobs and immutable downloadable generations."""
import os
from typing import Literal
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from .storage import ROOT, identifier, read_json


class PianoRequest(BaseModel):
    input: Literal['mix','piano']='mix'
    bpm: float | None = Field(default=None,ge=20,le=300)
    first_downbeat: float | None = Field(default=None,ge=0,le=1200)
    beats_per_bar: Literal[2,3,4,5,6,7,8,9,12] | None = None

    @model_validator(mode='after')
    def complete_grid(self):
        fields=[self.bpm,self.first_downbeat,self.beats_per_bar]
        if any(v is not None for v in fields) and not all(v is not None for v in fields):
            raise ValueError('手动节拍需要同时填写速度、小节起点和每小节拍数')
        return self


def register_piano(app,store):
    from .piano_worker import latest, FILES, PIANO_PYTHON

    @app.post('/api/projects/{project_id}/piano',status_code=202)
    def create(project_id:str,body:PianoRequest):
        if os.environ.get('HARMONY_EDITOR_PREVIEW')=='1':
            raise HTTPException(403,'预览环境不运行音频模型')
        project=store.project(project_id)
        if not project.get('audio'):
            raise HTTPException(400,'请先上传音频')
        if not PIANO_PYTHON.is_file():
            raise HTTPException(503,'钢琴转写环境尚未准备好')
        if project.get('active_job'):
            active=read_json(store.job_path(project['active_job']))
            if active.get('profile')!='piano' or active.get('pipeline',{})!=body.model_dump(exclude_none=True):
                raise HTTPException(409,'当前项目还有任务在运行，请完成后再生成钢琴 MIDI')
        return app.state.jobs.submit(project_id,'piano',pipeline=body.model_dump(exclude_none=True))

    @app.get('/api/projects/{project_id}/piano')
    def result(project_id:str):
        folder,manifest=latest(store,project_id)
        prefix=f'/api/projects/{project_id}/piano/{manifest["generation"]}/'
        manifest['urls']={name:prefix+name for name in manifest['files']}
        manifest['urls']['manifest.json']=prefix+'manifest.json'
        return manifest

    @app.get('/api/projects/{project_id}/piano/{generation}/{name}')
    def download(project_id:str,generation:str,name:str):
        identifier(generation);store.project(project_id)
        if name not in FILES:
            raise HTTPException(404,'未知文件')
        folder=store.project_dir(project_id)/'piano'/generation
        manifest=read_json(folder/'manifest.json')
        if name!='manifest.json' and name not in manifest['files']:
            raise HTTPException(404,'该版本没有此文件')
        media='audio/midi' if name.endswith('.mid') else 'audio/wav' if name.endswith('.wav') else 'application/json'
        return FileResponse(folder/name,media_type=media,filename=name,
                            content_disposition_type='attachment' if name.endswith('.mid') else 'inline')
