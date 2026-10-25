import hashlib
import json
from copy import deepcopy

import pytest

from tools.score_blind_boundary_review import points, validate, score, compare_objects, run


def sample():
    key = [dict(id=f'B{i}', source={'sha256':'abc'}) for i in range(2)]
    obj = dict(kind='complete_droplet', points=[[100, 200], [150, 200]])
    rows = [dict(id=f'B{i}', review_complete=True, objects=[deepcopy(obj)],
                 upper_wall_visibility='invisible', lower_wall_visibility='uncertain',
                 upper_wall_points=None, lower_wall_points=None, note='') for i in range(2)]
    a = dict(schema='blind-boundary-review-v1', package_id=hashlib.sha256(b'abcabc').hexdigest(),
             coordinate_system='original_sensor_px', reviewer='synthetic', human_review_complete=True, images=rows)
    return a, deepcopy(a), key


def test_perfect_repeats_never_authorize_measurement_or_strip_threshold():
    a,p,k=sample();r=score(a,p,k)
    assert r['matched_class_agreement']==1
    assert r['complete_droplet_endpoint_p90_sensor_px']==0
    assert r['engineering_threshold_passed'] is None
    assert r['measurement_accepted'] is False


@pytest.mark.parametrize('bad', [[[True,0],[20,0]], [[float('nan'),0],[20,0]], [[-1,0],[20,0]], [[10,540],[20,0]], [[10,0],[10,20]]])
def test_bad_coordinates_rejected(bad):
    with pytest.raises(ValueError): points(bad)


def test_split_merge_not_forced_into_matching():
    a=[dict(kind='complete_droplet',points=[[10,20],[100,20]])]
    b=[dict(kind='complete_droplet',points=[[10,20],[40,20]]),dict(kind='gap',points=[[60,20],[100,20]])]
    r=compare_objects(a,b)
    assert r['uniquely_matched']==0 and r['unmatched_or_ambiguous_second']==2
    assert r['count_changed'] is True


def test_class_mismatch_retained_not_used_as_endpoint_agreement():
    a,p,k=sample();a['images'][1]['objects'][0]['kind']='gap';r=score(a,p,k)
    assert r['matched_class_agreement']==0
    assert r['endpoint_sample_count']==0


def test_incomplete_answers_do_not_open_key(tmp_path):
    a,p,k=sample();a.update(human_review_complete=False,reviewer=None)
    for r in a['images']:r['review_complete']=False
    (tmp_path/'blank_answers.json').write_text(json.dumps(p),encoding='utf-8')
    path=tmp_path/'answers.json';path.write_text(json.dumps(a),encoding='utf-8')
    # No key exists: pending path must still succeed.
    assert run(path,tmp_path,tmp_path/'result.json')['status']=='human_review_pending'


def test_claimed_complete_empty_or_wall_missing_rejected():
    a,p,k=sample();a['images'][0]['objects']=[]
    with pytest.raises(ValueError,match='note'):validate(a,p)
    a,p,k=sample();a['images'][0]['lower_wall_visibility']='visible'
    with pytest.raises(ValueError,match='points'):validate(a,p)


def test_mismatched_package_and_key_rejected():
    a,p,k=sample();a['package_id']='wrong'
    with pytest.raises(ValueError,match='package'):validate(a,p)
    a,p,k=sample();k[0]['source']['sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):score(a,p,k)
