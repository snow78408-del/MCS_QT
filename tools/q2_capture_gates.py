"""Bounded operator readiness and pre-transaction prediction gates for Q2 capture."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from tools.plant_flow_transient_capture import CaptureLifecycleError
from tools.q2_prediction_gate import atomic_json, finite_number, json_bytes, load_bundle_for_plan, predict


def validate_gates(plan: dict) -> list[str]:
    if 'stage_gates' not in plan:
        return []  # Legacy capture plans retain their timing, explicitly documented.
    gates = plan['stage_gates']
    try:
        if gates.get('schema_version') != 1:
            raise ValueError('Unsupported stage gate schema')
        ready = gates['readiness']; prediction = gates['prediction']
        if ready.get('mode') != 'operator':
            raise ValueError('Readiness requires explicit operator confirmation')
        remind, limit = ready['remind_after_s'], ready['maximum_wait_s']
        if not all(finite_number(v) for v in (remind, limit)) or not 0 < remind <= limit <= 90:
            raise ValueError('Readiness wait must be bounded: 0 < reminder <= maximum <= 90s')
        if gates.get('role') not in ('training', 'sham', 'validation'):
            raise ValueError('Explicit training/sham/validation role required')
        if not isinstance(prediction.get('required'), bool):
            raise ValueError('Prediction required must be a boolean')
        if gates['role'] == 'validation' and not prediction['required']:
            raise ValueError('Validation cannot bypass prediction saving')
        if prediction['required']:
            timeout = prediction['timeout_s']
            if not finite_number(timeout) or not 0 < timeout <= 30:
                raise ValueError('Prediction timeout must be in (0,30] seconds')
            if plan.get('capture_plan', {}).get('raw_only') is True:
                raise ValueError('Prediction requires registered geometry, not raw-only capture')
            load_bundle_for_plan(plan)
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as error:
        return ['Stage gates: '+str(error)]
    return []


class Q2GateMixin:
    """Uses the existing monitored wait and existing finally/verified-stop lifecycle."""

    def _prepare_provenance(self) -> None:
        self._prepare_stage_gates()
        super()._prepare_provenance()

    def _prepare_stage_gates(self) -> None:
        if getattr(self, '_gates_prepared', False) or 'stage_gates' not in self.plan:
            return
        errors = validate_gates(self.plan)
        if errors:
            raise CaptureLifecycleError('; '.join(errors))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        probe = self.output_dir/'gate_write_probe.json'
        atomic_json(probe, dict(probe=True)); probe.unlink()
        if self.plan['stage_gates']['prediction']['required']:
            blob, self._frozen_bundle = load_bundle_for_plan(self.plan)
            # Pin exact bytes before pump connection. Later edits to the source cannot refit this session.
            with (self.output_dir/'frozen_prediction_bundle.json').open('xb') as stream:
                stream.write(blob); stream.flush(); os.fsync(stream.fileno())
            self._bundle_hash = hashlib.sha256(blob).hexdigest()
            self._frozen_model_identity = self._bundle_hash
        self._gates_prepared = True

    def _gate_poll(self, seconds: float) -> None:
        # STOP, camera/writer errors, storage and cumulative delivery are checked by this wait.
        self._wait_segment(seconds)
        if (self.output_dir/'PAUSE').exists():
            raise CaptureLifecycleError('Pause requested: stopping capture, no automatic resume')

    def _gate_reserve(self, *, before_first: bool = False, transition_started: float | None = None) -> None:
        self._update_delivery()
        before, after = self.stages
        remaining_transition = 60. if transition_started is None else 60.-(self._clock()-transition_started)
        if remaining_transition < 0:
            raise CaptureLifecycleError('Total transition preparation/transaction exceeds 60s reserve')
        for i, key in enumerate(('q1_ul_min', 'q2_ul_min')):
            old, new = before[key]+.5, after[key]+.5
            future = (old*5 if before_first else 0)+new*5+max(old,new)*remaining_transition/60+10
            if self._delivery_ul[i]+future > self._delivery_limit_ul:
                raise CaptureLifecycleError('Insufficient whole-block reserve while waiting; no counter reset')

    def _await_operator_ready(self) -> None:
        if 'stage_gates' not in self.plan:
            return
        self._prepare_stage_gates()
        config = self.plan['stage_gates']['readiness']; started = self._clock()
        self.sink.set_phase('awaiting_operator_ready', started)
        request = dict(session_id=self.plan['session_id'], nonce=secrets.token_hex(16),
            started_monotonic=started, deadline_monotonic=started+config['maximum_wait_s'],
            status='waiting',confirmation_required=['both_walls_visible', 'complete_droplet_endpoints_visible'])
        atomic_json(self.output_dir/'readiness_request.json', request)
        self._record('operator_readiness_wait_started')
        reminded = False
        try:
            while True:
                self._gate_poll(0.)
                self._gate_reserve(before_first=True)
                now = self._clock()
                if now-started >= config['maximum_wait_s']:
                    raise CaptureLifecycleError('Operator readiness maximum wait reached; raw data retained')
                response_path = self.output_dir/'readiness_confirmation.json'
                if response_path.exists():
                    response = json.loads(response_path.read_text(encoding='utf-8'))
                    stamp = response.get('confirmed_monotonic')
                    if (response.get('nonce') != request['nonce'] or response.get('session_id') != request['session_id']
                            or not finite_number(stamp) or not started <= stamp <= now
                            or not isinstance(response.get('operator'), str) or not response['operator'].strip()
                            or any(response.get(k) is not True for k in request['confirmation_required'])):
                        raise CaptureLifecycleError('Invalid/stale readiness confirmation')
                    atomic_json(self.output_dir/'readiness_accepted.json', dict(request=request,response=response,
                        accepted_monotonic=now,conservative_delivery_upper_ul=list(self._delivery_ul)))
                    self._record('operator_readiness_confirmed')
                    break
                if not reminded and now-started >= config['remind_after_s']:
                    self._record('operator_readiness_reminder_budget_rechecked'); reminded = True
                self._gate_poll(min(.25, config['maximum_wait_s']-(now-started)))
        finally:
            request['status'] = 'closed'
            atomic_json(self.output_dir/'readiness_request.json', request)

    def _launch_prediction_worker(self):
        self._prediction_log = (self.output_dir/'prediction_worker.log').open('xb')
        try:
            return subprocess.Popen([sys.executable, str(Path(__file__).with_name('q2_prediction_gate.py')),
                '--request',str(self.output_dir/'prediction_request.json')],
                stdout=self._prediction_log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except BaseException:
            self._prediction_log.close()
            raise

    def _save_prediction_before_switch(self, baseline: dict, transition_started: float) -> None:
        config = self.plan.get('stage_gates', {}).get('prediction', {})
        if not config.get('required'):
            return
        self._prepare_stage_gates()
        self._gate_poll(0.)
        self._gate_reserve(transition_started=transition_started)
        self.sink.set_phase('prediction_preparation', self._clock())
        geometry = json.loads((self.output_dir/'registered_geometry.json').read_text(encoding='utf-8'))
        # Snapshot only complete persisted NDJSON lines; the writer continues without a blocking drain.
        blob = (self.output_dir/'frame_manifest.ndjson').read_bytes()
        blob = blob[:blob.rfind(b'\n')+1]
        with (self.output_dir/'prediction_manifest_snapshot.ndjson').open('xb') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        request = dict(session_id=self.plan['session_id'],baseline_stage=baseline,
            registered_walls=geometry['walls'],bundle_sha256=self._bundle_hash,
            manifest_sha256=hashlib.sha256(blob).hexdigest(),
            q0=[self.stages[0][k] for k in ('q1_ul_min','q2_ul_min')],
            q1=[self.stages[1][k] for k in ('q1_ul_min','q2_ul_min')])
        request_path = self.output_dir/'prediction_request.json'; atomic_json(request_path,request)
        started = self._clock(); worker = self._launch_prediction_worker()
        try:
            while worker.poll() is None:
                self._gate_poll(.25)
                self._gate_reserve(transition_started=transition_started)
                if self._clock()-started >= config['timeout_s']:
                    raise CaptureLifecycleError('Prediction worker timeout; target not commanded')
            self._gate_poll(0.)
            if worker.returncode != 0:
                raise CaptureLifecycleError('Prediction preparation failed; see prediction_worker.log; target not commanded')
            candidate = json.loads((self.output_dir/'prediction_candidate.json').read_text(encoding='utf-8'))
            if (candidate.get('request_sha256') != hashlib.sha256(request_path.read_bytes()).hexdigest()
                    or candidate.get('bundle_sha256') != self._bundle_hash
                    or candidate.get('session_id') != self.plan['session_id']):
                raise CaptureLifecycleError('Prediction identity mismatch')
            expected = predict(self._frozen_bundle, candidate['baseline_bins'], request['q0'], request['q1'])
            if any(candidate.get(k) != v for k,v in expected.items()):
                raise CaptureLifecycleError('Prediction curves differ from frozen model')
            atomic_json(self.output_dir/'prospective_prediction.json', candidate)
            digest = hashlib.sha256((self.output_dir/'prospective_prediction.json').read_bytes()).hexdigest()
            atomic_json(self.output_dir/'prediction_seal.json',dict(prediction_sha256=digest,
                bundle_sha256=self._bundle_hash,saved_monotonic=self._clock(),target_command_sent=False))
            self._prediction_hash = digest
            self._record('prediction_saved_before_transition')
            self._gate_poll(0.)
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    worker.kill(); worker.wait(timeout=2)
            if getattr(self,'_prediction_log',None) is not None:
                self._prediction_log.close()
            self._update_delivery()

    def _verify_prediction_seal(self) -> None:
        if not self.plan.get('stage_gates',{}).get('prediction',{}).get('required'):
            return
        blob = (self.output_dir/'prospective_prediction.json').read_bytes()
        if hashlib.sha256(blob).hexdigest() != getattr(self,'_prediction_hash',None):
            raise CaptureLifecycleError('Prediction artifact changed before transaction')
