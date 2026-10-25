import hashlib
import json

import pytest

from tools.build_blind_boundary_review import build, repeated_order


def test_repeats_are_separated_and_balanced():
    result = repeated_order(24)
    assert sorted(result[:24]) == sorted(result[24:]) == list(range(24))
    assert all(result[24:].index(i)+24-result[:24].index(i) >= 10 for i in range(24))


def test_pack_preserves_originals_and_keeps_answers_blank(tmp_path):
    sources = []
    for i in range(12):
        p = tmp_path / f'secret_block_{i}.png'
        p.write_bytes(bytes([i]))
        sources.append(dict(image=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), block='SECRET'))
    out = tmp_path / 'pack'
    build(sources, out)
    public = json.loads((out / 'blank_answers.json').read_text(encoding='utf-8'))
    assert public['human_review_complete'] is False
    assert all(r['lower_wall_visibility'] is None and not r['objects'] for r in public['images'])
    html = (out / 'review.html').read_text(encoding='utf-8')
    assert 'SECRET' not in html and 'secret_block' not in html
    assert len(list((out / 'images').glob('*.png'))) == 24
    key = json.loads((out / 'key_keep_closed.json').read_text(encoding='utf-8'))
    for r in key:
        assert hashlib.sha256((out / 'images' / f'{r["id"]}.png').read_bytes()).hexdigest() == r['source']['sha256']
    with pytest.raises(ValueError, match='overwrite'):
        build(sources, out)


def test_changed_original_is_rejected_before_package_creation(tmp_path):
    sources = [dict(image=str(tmp_path / str(i)), sha256=str(i)) for i in range(12)]
    for s in sources:
        open(s['image'], 'wb').write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        build(sources, tmp_path / 'pack')
    assert not (tmp_path / 'pack').exists()
