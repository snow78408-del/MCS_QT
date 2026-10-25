import numpy as np
import pytest
from tools.audit_temporal_geometry import fit_wall


def test_slanted_ridge_recovered_despite_local_bright_object():
    im=np.zeros((120,300),float)
    for x in range(300):
        im[round(40+.03*x),x]=100
    im[48:51,130:140]=150
    seed=dict(x1=.1,x2=.9,y1=44/120,y2=52/120)
    wall,diag=fit_wall(im,seed)
    assert wall['y1']*120 == pytest.approx(40.9,abs=1)
    assert wall['y2']*120 == pytest.approx(48.1,abs=1)
    assert diag['retained']>30


def test_search_outside_image_rejected():
    with pytest.raises(ValueError):
        fit_wall(np.zeros((100,300)),dict(x1=.1,x2=.9,y1=0,y2=0))
