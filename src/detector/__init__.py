"""Real-footage smoke / fog detector for the far-field (one frame per minute) camera networks.

Appearance from a frozen CLIP image encoder (src/detector/features.py), linear
heads fitted on labelled real windows, thresholds fixed on training cameras
(src/detector/model.py). Built because the CFD-pretrained motion expert alone
does not transfer to real fog; its motion features can be added as an extra
input group.
"""
