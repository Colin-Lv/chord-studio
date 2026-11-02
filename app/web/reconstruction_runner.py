"""Isolated frozen-model adapters. No chord-conditioned note generation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from app.runtime_paths import configured_path, models_root
MODELS = configured_path('reconstruction_models', 'CHORD_RECONSTRUCTION_MODELS', models_root()/'reconstruction')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=['basic', 'drums', 'yourmt3'])
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.source = args.source.resolve(); args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    started = time.monotonic()
    settings = {}
    if args.model == 'basic':
        import basic_pitch
        from basic_pitch.inference import Model, predict
        model_path = Path(basic_pitch.__file__).parent/'saved_models/icassp_2022/nmp.onnx'
        settings = dict(onset_threshold=.5, frame_threshold=.3, minimum_note_length=80,
                        multiple_pitch_bends=False)
        _, midi, events = predict(str(args.source), Model(str(model_path)), **settings)
        midi.write(str(args.output))
        # Keep model bend contours too: MIDI cannot express independent bends
        # on overlapping notes on the same channel.
        args.output.with_suffix('.events.json').write_text(json.dumps(events, default=lambda x: x.tolist()), encoding='utf8')
    elif args.model == 'drums':
        import soundfile as sf
        import torch
        torch.set_num_threads(2)
        bundle = MODELS/'adt-str'
        sys.path.insert(0, str(bundle))
        from adt_transcriber import ADTTranscriber
        model = ADTTranscriber.from_pretrained(str(bundle), variant='setting-tau-0.8', device='cuda' if torch.cuda.is_available() else 'cpu')
        samples, sr = sf.read(args.source, dtype='float32', always_2d=True)
        result = model.transcribe(torch.from_numpy(samples.mean(axis=1)), sample_rate=sr,
                                  output_dir=str(args.output.parent/'adt_raw'), batch_size=2)
        import shutil
        shutil.copyfile(result, args.output)
        # ADT-STR's decoder emits a reduced custom drum vocabulary, not GM.
        # Convert using the author's published inverse before any GM playback.
        import pretty_midi
        from utils.mapping_utils import MappingUtils
        mapping=MappingUtils().GM_custom_midi_to_Gm_standard_Mapping
        midi=pretty_midi.PrettyMIDI(str(args.output))
        for inst in midi.instruments:
            for note in inst.notes:
                if note.pitch not in mapping:raise ValueError(f'Unknown ADT drum class {note.pitch}')
                note.pitch=int(mapping[note.pitch])
        midi.write(str(args.output))
        settings = dict(variant='setting-tau-0.8', batch_size=2, device=model.device.type if hasattr(model.device, 'type') else str(model.device))
        settings.update(drum_mapping='author custom-to-GM representative mapping',offset_policy='fixed 100 ms trigger; not acoustic duration')
    else:
        import numpy as np
        import torch
        import soundfile as sf
        import torchaudio
        torch.set_num_threads(2)
        bundle = MODELS/'yourmt3'
        sys.path[:0] = [str(bundle), str(bundle/'amt/src')]
        os.chdir(bundle)
        from model_helper import load_model_checkpoint
        from utils.audio import slice_padded_array
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from utils.note2event import mix_notes
        from utils.utils import write_model_output_as_midi
        checkpoint = 'mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt'
        options = [checkpoint, '-p','2024','-tk','mc13_full_plus_256','-dec','multi-t5','-nl','26',
                   '-enc','perceiver-tf','-sqr','1','-ff','moe','-wf','4','-nmoe','8','-kmoe','2',
                   '-act','silu','-epe','rope','-rp','1','-ac','spec','-hop','300','-atc','1','-pr','32']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model_checkpoint(options, device='cpu')
        checkpoint_path=bundle/'amt/logs/2024'/checkpoint.split('@')[0]/'checkpoints/last.ckpt'
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)['state_dict']
        expected={k for k in model.state_dict() if 'pitchshift' not in k}
        actual={k for k in state if 'pitchshift' not in k}
        if actual!=expected:raise RuntimeError(f'YourMT3 checkpoint keys differ: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
        del state
        model=model.to(device)
        samples, sr = sf.read(args.source, dtype='float32', always_2d=True)
        rate = model.audio_cfg['sample_rate']; frames = model.audio_cfg['input_frames']
        audio = torchaudio.functional.resample(torch.from_numpy(samples.mean(axis=1)),sr,rate).numpy()
        segments = torch.from_numpy(slice_padded_array(audio[None,:],frames,frames).astype('float32')).unsqueeze(1).to(device)
        with torch.inference_mode():
            tokens, _ = model.inference_file(bsz=1, audio_segments=segments)
        starts = [frames*i/rate for i in range(len(segments))]
        channels = []
        for ch in range(model.task_manager.num_decoding_channels):
            zipped, _, _ = model.task_manager.detokenize_list_batches([a[:,ch,:] for a in tokens], starts, return_events=True)
            notes, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
            channels.append(notes)
        write_model_output_as_midi(mix_notes(channels), str(args.output.parent), args.output.stem,
                                   model.midi_output_inverse_vocab)
        made = args.output.parent/'model_output'/args.output.name
        if made.exists():
            import shutil
            shutil.copyfile(made,args.output)
        settings = dict(options=options, device=device, batch_size=1,checkpoint_keys_verified=True)
    from .storage import signature
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(args.output))
    report = dict(model=args.model, settings=settings, input=signature(args.source),
                  notes=sum(len(i.notes) for i in midi.instruments), elapsed_seconds=time.monotonic()-started)
    bundle = MODELS/('adt-str' if args.model=='drums' else 'yourmt3')
    if args.model != 'basic': report['download_manifest']=json.loads((bundle/'download-manifest.json').read_text(encoding='utf8'))
    else: report['weights']=signature(model_path)
    args.output.with_suffix('.inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'download_manifest','input'}},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
