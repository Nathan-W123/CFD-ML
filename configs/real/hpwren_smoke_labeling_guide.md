# Labelling guide: smoke in HPWREN windows

Each contact sheet shows one 11-minute window from a fixed wildfire-watch
camera in Southern California, recording one frame per minute:

* top row: the full frames at t = 0 (left) and t = 11 min (right);
* bottom row: the same two frames, zoomed in on a region of interest.

The camera's banner (name, date, time) is blanked. Sheets come from
windows with and without fires, and the sheet does not tell you which. Judge
only what you see, and label each sheet on its own. Compare the two frames:
a new or growing plume is strong evidence.

## smoke: 1, 0 or unknown

**1** when smoke from a fire is visible in either frame, in the full frame
or the zoom:

* a plume or column rising from terrain, even a thin or small one;
* a drifting smoke cloud with its base on the ground;
* a smoke pall over a ridge that is clearly fed from below.

Smoke from a fire is usually grey, brown or white, wispy, and anchored to a
point on the ground, and it often widens and leans with the wind.

**0** when you are confident there is no smoke from a fire anywhere in view.
These are 0:

* clouds detached from the ground;
* fog or low stratus layers;
* haze bands along the horizon;
* dust from roads;
* sun glare and lens flare.

**unknown** when you cannot tell:

* a faint grey wisp that could be smoke or cloud;
* a brownish layer that could be smoke haze or smog;
* the region is too dark or washed out;
* water drops on the lens.

Unknown is better than a guess.

## smoke_where

`zoom` (visible in the zoomed region), `elsewhere` (only outside it),
`both`, or `none` when smoke is not 1.

## fog: 1, 0 or unknown

* **1:** fog or low cloud at ground level is visible: a fog or stratus sea
  below the camera, fog in valleys, a fog bank, low cloud resting on ridges,
  or the camera inside fog.
* **0:** none of these. Haze bands and clouds up in the sky are 0.
* **unknown:** cannot tell.

## quality

| value | meaning |
|---|---|
| `ok` | usable |
| `dark` | too dark to judge |
| `obstructed` | water drops or dirt on the lens hide the scene, the frame is corrupt, or there is a "no image" placeholder |
| `glare` | sun glare hides much of the view |

## confidence and note

* **confidence:** `high`, `medium` or `low`, for the smoke label.
* **note:** a few words, for example "thin grey plume rising from ridge at
  right, grows by t=11", "no smoke, cumulus above", or "haze band only".
