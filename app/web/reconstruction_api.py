"""Versioned project reconstruction and strictly scoped downloads."""
import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .storage import identifier,read_json


class ReconstructionRequest(BaseModel):
    comparison:bool=True


def register_reconstruction(app,store):
    from .reconstruction_worker import latest,PYTHON
    @app.post('/api/projects/{project_id}/reconstruction',status_code=202)
    def create(project_id:str,body:ReconstructionRequest):
        if os.environ.get('HARMONY_EDITOR_PREVIEW')=='1':raise HTTPException(403,'预览环境不运行音频模型')
        p=store.project(project_id)
        if not p.get('audio'):raise HTTPException(400,'请先上传歌曲')
        if not PYTHON.is_file():raise HTTPException(503,'整曲重构环境尚未安装')
        if p.get('active_job'):
            job=read_json(store.job_path(p['active_job']))
            if job['profile']!='reconstruction' or job.get('pipeline')!=body.model_dump():
                raise HTTPException(409,'当前项目还有任务，请完成后再生成')
        return app.state.jobs.submit(project_id,'reconstruction',pipeline=body.model_dump())

    @app.get('/api/projects/{project_id}/reconstruction')
    def result(project_id:str):
        _,manifest=latest(store,project_id)
        prefix=f'/api/projects/{project_id}/reconstruction/{manifest["generation"]}/'
        manifest['urls']={name:prefix+name for name in [*manifest['files'],'manifest.json']}
        return manifest

    @app.get('/api/projects/{project_id}/reconstruction/{generation}/{name}')
    def download(project_id:str,generation:str,name:str):
        identifier(generation);store.project(project_id)
        folder=store.project_dir(project_id)/'reconstruction'/generation
        manifest=read_json(folder/'manifest.json')
        if name!='manifest.json' and name not in manifest['files']:raise HTTPException(404,'未知文件')
        if '/' in name or '\\' in name or name.startswith('.'):raise HTTPException(404,'未知文件')
        media='audio/midi' if name.endswith('.mid') else 'audio/wav' if name.endswith('.wav') else 'application/zip' if name.endswith('.zip') else 'application/json' if name.endswith('.json') else 'text/plain'
        return FileResponse(folder/name,media_type=media,filename=name,
                            content_disposition_type='attachment' if name.endswith(('.zip','.mid')) else 'inline')
