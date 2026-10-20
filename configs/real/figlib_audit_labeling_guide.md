# Labelling guide: any smoke in view (wildfire-camera windows)

Each contact sheet shows one 11-minute window from a fixed wildfire-watch
camera in Southern California, recording one frame per minute: four full
frames, at t = 0, 4, 8 and 11 min, in a 2 x 2 grid. The camera's banner
(name, date, time) is blanked. Some footage is old, low-contrast, or
grayscale (night infrared). Judge only what you see, and label each sheet on
its own; the sheet does not tell you whether a fire is nearby.

## smoke: 1, 0 or unknown

**1** when smoke from a fire (wildfire, burn, structure fire) is visible
anywhere in any of the four frames:

* a plume or column rising from terrain, even a thin or small one;
* a drifting smoke cloud with its base on the ground;
* a smoke pall over a ridge that is clearly fed from below.

Smoke from a fire is usually grey, brown or white, wispy, and anchored to a
point on the ground, and it often widens and leans with the wind. A plume
that appears or grows between frames is strong evidence.

**0** when you are confident there is no smoke from a fire anywhere in view.
These are 0: clouds detached from the ground; fog or low stratus layers;
haze bands along the horizon; dust from roads; sun glare and lens flare.

**unknown** when you cannot tell: a faint grey wisp that could be smoke or
cloud; a brownish layer that could be smoke haze or smog; the view too dark
or washed out; water drops on the lens. Unknown is better than a guess.

## smoke_where

`elsewhere` when smoke is 1, `none` otherwise (there is no zoomed region on
these sheets; do not use `zoom` or `both`).

## fog: 1, 0 or unknown

* **1:** fog or low cloud at ground level is visible (a fog or stratus sea
  below the camera, fog in valleys, a fog bank, low cloud on ridges, or the
  camera inside fog).
* **0:** none of these. Haze bands and clouds in the sky are 0.
* **unknown:** cannot tell.

## quality

`ok` usable; `dark` too dark to judge; `obstructed` lens drops or dirt,
corrupt frame or placeholder; `glare` sun glare hides much of the view.

## confidence and note

**confidence:** `high`, `medium` or `low`, for the smoke label. **note:** a
few words, e.g. "thin grey plume rising from ridge at left, grows by t=11",
"no smoke, cumulus above", "haze band only".
