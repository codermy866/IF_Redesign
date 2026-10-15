"""Protect equal-dimensional zero-origin RIME readout comparisons."""
import torch
from if_redesign.rime.readout import coordinates


def test_moment_laplace_both_zero_at_origin():
    zero = torch.zeros((2, 3, 8), dtype=torch.float64)
    for kind in ("moment", "laplace"):
        phi = coordinates(zero, kind)
        assert phi.shape == (2, 3, 56)
        torch.testing.assert_close(phi, torch.zeros_like(phi), atol=1e-12, rtol=0)


def test_moment_matches_legendre_zero_anchor():
    values = torch.linspace(-1, 1, 21, dtype=torch.float64).reshape(21,1,1).expand(21,1,8)
    phi = coordinates(values, "moment").reshape(21,1,8,7)
    x = values
    p0, p1 = torch.ones_like(x), x
    at_zero = [1.,0.]
    expect = [p1]
    for n in range(2,8):
        pn = ((2*n-1)*x*p1-(n-1)*p0)/n
        zn = -(n-1)*at_zero[n-2]/n
        at_zero.append(zn)
        expect.append(pn-zn)
        p0,p1=p1,pn
    ref = torch.stack(expect,dim=-1)
    torch.testing.assert_close(phi,ref,atol=1e-12,rtol=1e-12)
