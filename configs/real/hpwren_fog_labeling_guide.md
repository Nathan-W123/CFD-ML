# Labelling guide: HPWREN fog windows

Each contact sheet shows one 11-minute window from a fixed wildfire-watch
camera in Southern California (one frame per minute): frames t = 0, 4, 8 and
11 min in a 2 x 2 grid. The camera's banner (name, date, time) is blanked.
Sheets were chosen from mornings with and without fog nearby, but the sheet
tells you nothing about which; judge only what you see. Label each sheet
independently.

## fog: 1, 0 or unknown

**1** when fog or low cloud at ground level is visible anywhere in the view,
in at least one frame:

* a fog or stratus sea below the camera, or fog filling valleys or basins
  (`below`);
* a fog bank at a distance, at about the camera's level or rolling over
  terrain (`bank`);
* the camera itself inside fog: the view is washed out grey or white and the
  terrain is hidden or faint (`enveloping`);
* small fog patches or wisps on slopes or in hollows (`patches`);
* low cloud whose base rests on ridges or peaks, hiding them (`low_cloud_on_terrain`).

**0** when no fog or low cloud touches the terrain anywhere in view. These
are 0:

* clear views;
* a haze or smog band along the horizon without a defined white or grey fog
  layer;
* clouds that are clearly up in the sky, such as cumulus, altocumulus,
  cirrus, or an overcast deck well above the terrain;
* rain shafts and virga.

**unknown** when you cannot tell:

* too dark;
* lens covered by water or dirt, so the scene is unreadable;
* sun glare washing out the area that matters;
* a thin layer that might be fog or might be haze;
* a cloud whose base may or may not touch distant terrain.

Unknown is better than a guess.

## fog_type and fog_extent (only when fog = 1; otherwise `none`)

* **fog_type:** the dominant one of `below`, `bank`, `enveloping`, `patches`
  or `low_cloud_on_terrain`.
* **fog_extent:** the largest image area fog covers in any frame: `small`
  (under 10 %), `medium` (10-40 %) or `large` (over 40 %).

## smoke: 1, 0 or unknown

* **1:** smoke from a fire is visible, such as a plume, a column, or smoke
  haze that clearly comes from a fire.
* **0:** you are confident there is none.
* **unknown:** a grey wisp that could be smoke or cloud, or brownish haze
  that could be smoke.

## quality

| value | meaning |
|---|---|
| `ok` | usable |
| `dark` | too dark to judge |
| `obstructed` | water drops or dirt on the lens hide the scene, the frame is corrupt, or the camera moved |
| `glare` | sun glare hides much of the view |

## confidence and note

* **confidence:** `high`, `medium` or `low`, for the fog label.
* **note:** a few words, for example "fog sea below, thinning", "haze band
  only", "cumulus above ridges, no fog", or "drops on lens".
