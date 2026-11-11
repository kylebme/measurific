"""Small, self-describing inference checkpoints without optimizer state."""
from pathlib import Path
import hashlib
import json
import torch
from .data import LABEL_VERSION,NAMES
from .train import load_model


def export_model(checkpoint,output,tokens=1024):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    model,ckpt=load_model(checkpoint)
    if ckpt.get('label_version') not in (LABEL_VERSION,'primitives-compounds-v1','primitives-compounds-v2'):raise ValueError('Unsupported checkpoint label version')
    if tokens<32:raise ValueError('Token budget must be at least 32')
    artifact={key:ckpt[key] for key in ('model','model_config','step','label_version','instances_trained')}
    artifact['settings']={'tokens':tokens}
    artifact['input_contract']={'coordinates':'actual_xy_mm: float64 [N,2]; use --units px for pixels',
        'connectivity':'ring_offsets: int64 [R+1], starts at 0 and ends at N',
        'closure':'implicit final edge back to first vertex; do not repeat the first point',
        'topology':'one outer closed ring and non-touching, non-nested through holes',
        'semantic_names':NAMES,'default_threshold':.7,'confidence_calibrated':False,
        'required_token_budget':tokens}
    if model.config.adaptive_sampling:
        artifact['settings']={'tokens':None,'sampling':model.config.sampling_version}
        artifact['input_contract'].pop('required_token_budget')
        artifact['input_contract'].update(sampling=model.config.sampling_version,max_intervals=65536,instance_policy='closed-loop topology, semantic boundaries and learned repeat associations')
    if model.config.label_schema in ('primitives-compounds-v1','primitives-compounds-v2'):
        from .inference_primitives import base_names_for_schema,COMPOUND_NAMES
        artifact['input_contract'].update(semantic_names=base_names_for_schema(model.config.label_schema),compound_names=COMPOUND_NAMES,
            compound_policy='Most likely class for each detected compound; retain base instance references')
    artifact['provenance']={'source_checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'data_fingerprint':ckpt.get('data_fingerprint'),'source_step':ckpt['step']}
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+'.partial');torch.save(artifact,temp);temp.replace(output)
    metadata={k:v for k,v in artifact.items() if k!='model'}
    metadata['artifact_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    return output
