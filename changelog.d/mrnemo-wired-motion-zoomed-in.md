bump: minor

### Added

- A fifth choice on the Masked Source's `motion_reference`: `a video I wire, zoomed in`. The video on `motion_video` is shown in the tracked subject's box of each shot and not in the whole frame, by the layout `subject only, zoomed in` already uses (`video_mask.zoom_plan`: never more pixels than the whole frame at `motion_short_edge`, never a smaller subject), so a wired body mesh gives the subject more of the model's tokens for the same rows. Nothing is greyed: every pixel of the wired video inside a box is kept, and what is not a box (the band around one, a frame of a shot the subject is in no frame of) is filled with the wired video's own ground (`video_mask.wired_ground`, the frame's corners) where the built-in zoom fills with the lane's grey, so a mesh on black stays on black. A source that carries no boxes is refused. Appended to the combo, so the choices before it keep their place; the song node reports the zoom as it does for the built-in one. Reasoned and untested on a render when written.

### Fixed

- The kept conditioning's key did not hold the wired motion video (`window_keep.cond_key`): with `reuse_windows` on, another video wired on the same source would have been handed the conditioning made from the first. It is in the key now, for both wired choices, and zoomed in so are the boxes and the shot table. `reuse_windows` is off by default, so no render so far read a kept conditioning this way.

### Checks

- `bench/check_video_mask.py`: the zoomed wired reference is the wired video laid in the window's subject boxes; the subject's block is much more of a picture that is no larger; the fill is the wired video's ground and not grey, on a band and on a shot with no subject; the whole-frame choice is unchanged when the record carries boxes; the choice with no boxes and with nothing wired is refused; the choice is appended.
- `bench/check_window_keep.py`: the wired video is in the key for both choices; a moved box misses zoomed in and hits whole; the two choices do not share a key; the restated groups are `video_mask.py`'s.

Not changed, and told to its owner: `bench/patch_render_window.py` remakes a motion clip for the built-in zoom only, so a patch of a render that used the new choice does not reproduce its reference.
