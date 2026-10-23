"""Experimental image-only separation of two colored light strips.

This checks for a darker, less chromatic interval between the proposed strips.
It does not redefine the offline eligible population and is not enabled by default.
"""
import numpy as np

TOPOLOGY_RULE = {
    "version": "two-strip-dark-gap-probe-v1",
    "rows": 9, "interior_samples": 31, "support_min": .6,
    "chroma_ratio": .5, "brightness_ratio": .9,
    "cyan_green_tolerance": 10,
}


def separated_lamp_evidence(image, corners, rule=TOPOLOGY_RULE):
    c = np.asarray(corners, float)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        return {"accepted": False, "reason": "invalid_corners", "gap_support": 0.}
    t = np.linspace(.2, .8, rule["rows"])
    left = c[0]+t[:, None]*(c[1]-c[0])
    right = c[3]+t[:, None]*(c[2]-c[3])
    vector = right-left
    length = np.linalg.norm(vector, axis=1)
    direction = vector/np.maximum(length[:, None], 1e-6)

    def sample(points):
        xy = np.rint(points).astype(int)
        inside = ((xy[..., 0] >= 0) & (xy[..., 0] < image.shape[1])
                  & (xy[..., 1] >= 0) & (xy[..., 1] < image.shape[0]))
        xy[..., 0] = xy[..., 0].clip(0, image.shape[1]-1)
        xy[..., 1] = xy[..., 1].clip(0, image.shape[0]-1)
        pixels = image[xy[..., 1], xy[..., 0]].astype(float)
        red = pixels[..., 0]-np.maximum(pixels[..., 1], pixels[..., 2])
        blue = np.where(pixels[..., 2] >= pixels[..., 1]-rule["cyan_green_tolerance"],
                        pixels[..., 2]-pixels[..., 0],
                        pixels[..., 2]-np.maximum(pixels[..., 0], pixels[..., 1]))
        return np.maximum(red, blue), pixels.max(-1), inside

    offset = np.arange(-1, 2)
    lc, lv, li = sample(left[:, None]+offset[None, :, None]*direction[:, None])
    rc, rv, ri = sample(right[:, None]+offset[None, :, None]*direction[:, None])
    li_peak, ri_peak = np.argmax(np.where(li, lc, -1e6), 1), np.argmax(np.where(ri, rc, -1e6), 1)
    rows = np.arange(len(t))
    reference_chroma = np.minimum(lc[rows, li_peak], rc[rows, ri_peak])
    reference_brightness = np.minimum(lv[rows, li_peak], rv[rows, ri_peak])
    u = np.linspace(.2, .8, rule["interior_samples"])
    middle = left[:, None]+u[None, :, None]*vector[:, None]
    mc, mv, mi = sample(middle)
    gaps = ((mc < rule["chroma_ratio"]*reference_chroma[:, None])
            & (mv < rule["brightness_ratio"]*reference_brightness[:, None]) & mi)
    row_ok = gaps.any(1) & (reference_chroma >= 35) & (reference_brightness >= 80) & (length >= 3)
    support = float(row_ok.mean())
    accepted = support >= rule["support_min"]
    return {"accepted": accepted, "reason": "distinct_strips_supported" if accepted else "no_resolved_dark_gap_between_strips",
            "gap_support": support, "row_gap": row_ok.tolist(), "rule_version": rule["version"]}
