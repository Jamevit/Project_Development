"""Record A-Z sequences, including J/Z movement, with optional automatic batches."""
from pathlib import Path
from datetime import datetime
import argparse
import time
import uuid
import numpy as np
import cv2
from hand_sequences import HandReader, sequence_features

ROOT = Path(__file__).resolve().parent
ALPHABET = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def new_session():
    return datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]

def validate_clip(points, timestamps, sides, attempts, duration):
    if not sides or len(set(sides)) != 1:
        raise ValueError('Use the same single hand throughout. Retry.')
    if len(points) < 15 or len(points) / max(attempts, 1) < .85:
        raise ValueError('Too few clear hand frames. Retry.')
    if timestamps[-1] - timestamps[0] < duration * .8:
        raise ValueError('Keep your hand visible from GO until the end.')
    # Validate using the existing sequence preprocessing; raw timestamps are retained.
    return sequence_features(points, timestamps, sides[0])

def save_clip(root, session, label, points, timestamps, hand, features, duration,
              split='unspecified'):
    if label not in ALPHABET:
        raise ValueError('Choose one letter A-Z')
    folder = Path(root) / session
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (label + '-' + uuid.uuid4().hex + '.npz')
    # Exclusive creation prevents overwriting a previous recording.
    with path.open('xb') as stream:
        np.savez_compressed(stream, label=label, points=np.asarray(points, dtype=np.float32),
                            timestamps=np.asarray(timestamps, dtype=np.float64), hand=hand,
                            session=session, duration=duration, features=features, split=split,
                            feature_version='external-jz-xy32-v1', mirrored=True,
                            created_at=datetime.now().astimezone().isoformat())
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--alphabet', action='store_true', default=True, help=argparse.SUPPRESS)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train',
                        help='Dataset split for an alphabet recording session')
    parser.add_argument('--batch', type=int, default=0,
                        help='Automatically save this many valid clips per SPACE press (0: manual review)')
    parser.add_argument('--rest', type=float, default=2.0,
                        help='Rest seconds between batch attempts, before the ready countdown')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 5:
        parser.error('--seconds must be between 1 and 5')
    if not 0 <= args.batch <= 100:
        parser.error('--batch must be between 0 and 100')
    if not 0 <= args.rest <= 30:
        parser.error('--rest must be between 0 and 30')
    session = new_session()
    labels = ALPHABET
    output = ROOT/'az_recordings'/args.split
    counts = dict.fromkeys(labels, 0)
    target = 10 if args.split == 'train' else 5
    label, state, status = labels[0], 'idle', 'Choose a letter, then SPACE'
    print('Recording folder:', output/session, flush=True)
    samples, stamps, sides, replay, replay_times = [], [], [], [], []
    features, found, attempts, started, next_sample = None, None, 0, 0.0, 0.0
    remaining, resume_at, last_outcome = 0, 0.0, ''
    cap = cv2.VideoCapture(args.device)
    reader = None
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps and retry.')
        reader = HandReader(ROOT / 'hand_landmarker.task')
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames')
            now = time.monotonic()
            elapsed = now - started
            fresh = now >= next_sample
            if fresh:
                found = reader.read(frame)
                next_sample = now + 1/15
            view = cv2.resize(cv2.flip(frame, 1), (640, 480))
            if found is not None:
                for x, y in found[0]:
                    cv2.circle(view, (int(x*640/frame.shape[1]), int(y*480/frame.shape[0])), 3, (90,255,90), -1)
            if state == 'rest':
                status = f'{last_outcome} | Rest {max(0, resume_at-now):.1f}s'
                if now >= resume_at:
                    samples, stamps, sides, replay, replay_times = [], [], [], [], []
                    attempts, features = 0, None
                    started, elapsed, state = now, 0.0, 'capture'
            if state == 'capture':
                if elapsed < 1:
                    status = f'Ready {1-elapsed:.1f}s - prepare {label}'
                elif elapsed < 1 + args.seconds:
                    action = 'perform' if label in ('J', 'Z') else 'hold'
                    status = f'GO: {action} {label} | {1+args.seconds-elapsed:.1f}s'
                    if fresh:
                        attempts += 1
                        replay.append(view.copy())
                        replay_times.append(elapsed-1)
                        if found is not None:
                            samples.append(found[0].copy())
                            stamps.append(elapsed-1)
                            sides.append(found[1])
                else:
                    try:
                        features = validate_clip(samples, stamps, sides, attempts, args.seconds)
                    except ValueError as error:
                        if args.batch:
                            state, resume_at = 'rest', now + args.rest
                            last_outcome = 'Tracking incomplete; retrying'
                            status = last_outcome
                            print('Not saved:', error, flush=True)
                        else:
                            state, status = 'idle', str(error)
                    else:
                        if args.batch:
                            path = save_clip(output, session, label, samples, stamps, sides[0],
                                             features, args.seconds,
                                             args.split)
                            counts[label] += 1
                            remaining -= 1
                            print('Saved:', path, flush=True)
                            if remaining:
                                state, resume_at = 'rest', now + args.rest
                                last_outcome = f'Saved {label} ({args.batch-remaining}/{args.batch})'
                                status = last_outcome
                            else:
                                state = 'idle'
                                status = f'Batch complete: {args.batch} saved. Choose a letter + SPACE.'
                        else:
                            state = 'review'
                            review_start = now
                            status = 'Review: S save | R retry | ESC exit'
            panel = np.zeros((620, 640, 3), dtype=np.uint8)
            panel[140:] = view
            if state == 'review' and replay:
                offset = (now-review_start) % args.seconds
                index = int(np.clip(np.searchsorted(replay_times, offset, side='right')-1,0,len(replay)-1))
                panel[140:] = replay[index]
            title = f'A-Z {args.split.upper()}'
            controls = 'A-Z: select | SPACE: record | Shift+8: new session'
            if args.batch:
                controls = 'SPACE: start/stop batch | ESC: exit'
                if state == 'idle':
                    controls = 'A-Z: select' + ' | ' + controls
            progress = f'Session: {sum(counts.values())} saved | {sum(n >= target for n in counts.values())}/26 letters at target {target}'
            lines = [f'{title} RECORDING | Selected: {label} | Saved here: {counts[label]}',
                     controls,
                     status,
                     (f'AUTO SAVE | Batch: {args.batch-remaining}/{args.batch} | Repeat selected sign on GO'
                      if args.batch and state in ('capture', 'rest') else
                      f'AUTO SAVE: {args.batch} clips per batch | Tracking checks only'
                      if args.batch else
                      'REPLAY: S save / R retry' if state == 'review' else 'One hand visible | Each clip is reviewed before saving'),
                     progress]
            for i, line in enumerate(lines):
                cv2.putText(panel,line,(10,23+i*25),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1,cv2.LINE_AA)
            cv2.imshow('SignBridge sequence recorder',panel)
            key=cv2.waitKey(1)&255
            if key==27: break
            if state in ('capture', 'rest'):
                if args.batch and key == 32:
                    state, remaining = 'idle', 0
                    status = 'Batch stopped. Saved clips kept. SPACE starts a new batch.'
                continue
            if state == 'review':
                if key in (ord('s'),ord('S')):
                    path=save_clip(output,session,label,samples,stamps,sides[0],features,args.seconds,
                                   args.split)
                    counts[label]+=1
                    print('Saved:',path,flush=True)
                    state,status='idle',f'Saved {label}. SPACE for another clip.'
                elif key in (ord('r'),ord('R')):
                    state,status='idle','Discarded unsaved clip. SPACE to retry.'
                continue
            if ord('a') <= key <= ord('z'): label=chr(key).upper()
            elif ord('A') <= key <= ord('Z'): label=chr(key)
            if key==ord('*'):
                session,counts=new_session(),dict.fromkeys(labels,0)
                status='New session. SPACE to record.'
            if key==32:
                samples,stamps,sides,replay,replay_times=[],[],[],[],[]
                attempts,features=0,None
                remaining = args.batch
                started,state=time.monotonic(),'capture'
    finally:
        if reader is not None: reader.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
